import pandas as pd
import subprocess
import math
import os
from src.config import *
from src.model import *
from src.type import *

num_call_trace = 0


class Ramulator:
    def __init__(
        self,
        modelinfos,
        ramulator_dir,
        output_log="",
        fast_mode=False,
        num_hbm=5,
        sparsity=False,
        row_size=16,
        sparsity_ratio=0.25,
        kv_budget=0,
    ):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)

        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        self.nhead = modelinfos["num_heads"]
        self.dhead = modelinfos["dhead"]
        self.fast_mode = fast_mode

        # Sparsity-related parameters (trace generation only).
        self.sparsity = sparsity
        self.row_size = row_size
        self.sparsity_ratio = sparsity_ratio
        self.kv_budget = kv_budget

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += "      channel: 16\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: HBM3_5.2Gbps\n"
        else:
            line += "      preset: HBM3_5.2Gbps_NPC\n"
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: HBM3-PIM\n"
        with open(yaml_file, "w") as f:
            f.write(line)

    def update_log_file(self, log):
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                columns = [
                    "L",
                    "nhead",
                    "dhead",
                    "dbyte",
                    "pim_type",
                    "power_constraint",
                    "cycle",
                    "mac",
                    "softmax",
                    "mvgb",
                    "mvsb",
                    "wrgb",
                ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df

        if len(df.columns) > 12:
            import pdb

            pdb.set_trace()

        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    def run_ramulator(
        self,
        pim_type: PIMType,
        l,
        num_ops_per_hbm,
        dbyte,
        yaml_file,
        file_name,
        l_target=0,
    ):
        pim_type_name = pim_type.name.lower() if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + ".trace")

        # Per-step KV budget: dict overrides scalar (scalar may be 0).
        kv_budget = getattr(self, "kv_budget_dict", {}).get(l, getattr(self, "kv_budget", 0))
        sparsity = getattr(self, "sparsity", False)
        row_size = getattr(self, "row_size", 16)
        sparsity_ratio = getattr(self, "sparsity_ratio", 0.25)

        trace_exc = os.path.join(
            self.ramulator_dir, f"trace_gen/gen_trace_attacc_{pim_type_name}.py"
        )
        trace_args = (
            f"--dhead {self.dhead} --nhead {num_ops_per_hbm} "
            f"--seqlen {l} --dbyte {dbyte} --output {trace_file}"
        )

        # Trace-gen flags for sparsity / token selection.
        if sparsity:
            trace_args += " --sparsity"
            if kv_budget > 0:
                trace_args += f" --kv_budget {kv_budget}"
            else:
                trace_args += f" --sparsity_ratio {sparsity_ratio}"
            if row_size != 16:
                trace_args += f" --row_size {row_size}"
                
            # When reproducing results other than STARC, comment out 
            # these two lines; the clustering overhead is not included.
            # -----------------------------------------
            if l == l_target - 1:
                trace_args += " --add_cluster"
            # -----------------------------------------
        else:
            if kv_budget > 0:
                trace_args += f" --kv_budget {kv_budget}"

        gen_trace_cmd = f"python {trace_exc} {trace_args}"

        # Generate trace.
        try:
            os.system(gen_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # Run ramulator.
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        try:
            result = subprocess.run(
                run_ramulator_cmd, stdout=subprocess.PIPE, text=True, shell=True
            )
            output_lines = result.stdout.strip().split("\n")
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # Remove trace.
        rm_trace_cmd = f"rm {trace_file}"
        try:
            os.system(rm_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # Parse ramulator output.
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        out = [cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"], n_cmds["wrgb"]]
        return out

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True, l_target=0):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            kv_budget = self.kv_budget_dict.get(l, self.kv_budget)

            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            file_name = (
                f"attacc_l{l}_kv{kv_budget}_nattn{num_ops_per_hbm}"
                f"_dhead{dhead}_dbyte{dbyte}_pc{int(power_constraint)}"
            )

            yaml_file = os.path.join(self.ramulator_dir, file_name + ".yaml")
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(
                pim_type, l, num_ops_per_hbm, layer.dbyte, yaml_file, file_name, l_target
            )

            rm_yaml_cmd = f"rm {yaml_file}"
            try:
                os.system(rm_yaml_cmd)
            except Exception as e:
                print(f"Error: {e}")

            # Post-processing: convert counts to traffic and latency.
            cycle, mac, sfm, mvgb, mvsb, wrgb = result
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 1

            log = [l, num_ops_per_hbm, dhead, dbyte, pim_type.name, power_constraint] + result
            self.update_log_file(log)

            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True, l_target=0):
        if self.df.empty:
            self.run(pim_type, layer, power_constraint, l_target=l_target)

        num_ops_per_attacc = layer.numOp
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        l = layer.n
        dhead = layer.k
        dbyte = layer.dbyte
        row = self.df[
            (self.df["L"] == l)
            & (self.df["nhead"] == num_ops_per_hbm)
            & (self.df["dbyte"] == dbyte)
            & (self.df["dhead"] == dhead)
            & (self.df["power_constraint"] == power_constraint)
            & (self.df["pim_type"] == pim_type.name)
        ]
        if row.empty:
            return self.run(pim_type, layer, power_constraint, l_target=l_target)
        else:
            cycle = int(row.iloc[0]["cycle"])
            mac = int(row.iloc[0]["mac"])
            softmax = int(row.iloc[0]["softmax"])
            mvgb = int(row.iloc[0]["mvgb"])
            mvsb = int(row.iloc[0]["mvsb"])
            wrgb = int(row.iloc[0]["wrgb"])
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 2

            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic
