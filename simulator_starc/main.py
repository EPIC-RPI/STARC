import argparse
import csv
import os
import pandas as pd

from src.system import *
from src.type import *
from src.config import *
from src.ramulator_wrapper import *

RAMULATOR = True


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, "a")
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                "model",
                "dtype",
                "xpu",
                "cap",
                "bw",
                "sys_opb",
                "hw",
                "cores",
                "pipe_level",
                "is parallel",
                "power constraint",
                "gqa_size",
                "Lin",
                "Lout",
                "bs",
                "required_cap",
                "s_flops",
                "g_flops",
                "s_time",
                "s_matmul",
                "s_fc",
                "s_comm",
                "s_softmax",
                "s_act",
                "s_lnorm",
                "g_time (ms)",
                "g_matmul",
                "g_fc",
                "g_comm",
                "g_etc",
                "g_qkv_time",
                "g_prj_time",
                "g_ff_time",
                "g2g_comm",
                "c2g_comm",
                "g_softmax",
                "g_act",
                "g_lnorm",
                "g_energy (nJ)",
                "g_dram_energy",
                "g_l2_energy",
                "g_l1_energy",
                "g_reg_energy",
                "g_alu_energy",
                "g_fc_mem_energy",
                "g_fc_comp_energy",
                "g_attn_mem_energy",
                "g_attn_comp_energy",
                "g_etc_mem_energy",
                "g_etc_comp_energy",
                "g_comm_energy",
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()


def read_kv_budget_from_file(file_path):
    """Read KV budget values from .txt/.csv/.xlsx/.xls (first column)."""
    budget_list = []

    if file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        budget_list.append(int(line))
                    except ValueError:
                        print(
                            f"Warning: Line {line_num} cannot be parsed as integer: '{line}'"
                        )

    elif file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Warning: CSV file {file_path} is empty")
                return budget_list

            for idx, value in enumerate(df.iloc[:, 0]):
                try:
                    if pd.notna(value) and str(value).strip():
                        budget_list.append(int(float(value)))
                except (ValueError, TypeError):
                    print(
                        f"Warning: CSV row {idx+2} column 1 cannot be parsed as integer: '{value}'"
                    )
        except Exception as e:
            print(f"Error: Failed to read CSV file {file_path}: {e}")
            return []

    elif file_path.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(file_path)
            if df.empty:
                print(f"Warning: Excel file {file_path} is empty")
                return budget_list

            for idx, value in enumerate(df.iloc[:, 0]):
                try:
                    if pd.notna(value) and str(value).strip():
                        budget_list.append(int(float(value)))
                except (ValueError, TypeError):
                    print(
                        f"Warning: Excel row {idx+2} column 1 cannot be parsed as integer: '{value}'"
                    )
        except Exception as e:
            print(f"Error: Failed to read Excel file {file_path}: {e}")
            return []

    else:
        print(f"Error: Unsupported file format: {file_path}")
        print("Supported formats: .txt, .csv, .xlsx, .xls")
        return []

    return budget_list


def write_excel(logfile, perfs):
    columns = [
        "model",
        "dtype",
        "xpu",
        "cap",
        "bw",
        "sys_opb",
        "hw",
        "cores",
        "pipe_level",
        "is parallel",
        "power constraint",
        "gqa_size",
        "Lin",
        "Lout",
        "bs",
        "required_cap",
        "s_flops",
        "g_flops",
        "s_time",
        "s_matmul",
        "s_fc",
        "s_comm",
        "s_softmax",
        "s_act",
        "s_lnorm",
        "g_time (ms)",
        "g_matmul",
        "g_fc",
        "g_comm",
        "g_etc",
        "g_qkv_time",
        "g_prj_time",
        "g_ff_time",
        "g2g_comm",
        "c2g_comm",
        "g_softmax",
        "g_act",
        "g_lnorm",
        "g_energy (nJ)",
        "g_dram_energy",
        "g_l2_energy",
        "g_l1_energy",
        "g_reg_energy",
        "g_alu_energy",
        "g_fc_mem_energy",
        "g_fc_comp_energy",
        "g_attn_mem_energy",
        "g_attn_comp_energy",
        "g_etc_mem_energy",
        "g_etc_comp_energy",
        "g_comm_energy",
    ]
    rows = []
    for perf in perfs:
        tag, config, time, energy = perf
        rows.append(tag + config + time + energy)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(logfile, index=False)


def run(
    system: System,
    batch,
    lin,
    lout,
    power_constraint=False,
    pipe=0,
    parallel=False,
    output_file=None,
):
    print(
        "---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".format(
            batch, lin, lout, pipe, parallel
        )
    )
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(
        batch,
        lin,
        lout,
        perfs=perfs,
        pipe=pipe,
        parallel_ff=parallel,
        power_constraint=power_constraint,
    )
    if output_file is not None:
        write_excel(output_file, perfs)


def main():
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--system",
        type=str,
        default="dgx",
        help="dgx, dgx-cpu, dgx-attacc",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="A100a",
        help="GPU type (A100a, H100)",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=8,
        help="number of GPUs in DGX system",
    )
    parser.add_argument(
        "--gmemcap",
        type=int,
        default=80,
        help="memory capacity per GPU (GB)",
    )

    parser.add_argument(
        "--pim",
        type=str,
        default="bank",
        help="pim mode: bank, bg, buffer",
    )
    parser.add_argument(
        "--powerlimit",
        action="store_true",
        help="power constraint for PIM",
    )
    parser.add_argument(
        "--ffopt",
        action="store_true",
        help="apply feedforward parallel optimization",
    )
    parser.add_argument(
        "--pipeopt",
        action="store_true",
        help="apply pipeline optimization",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GPT-175B",
        help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B",
    )
    parser.add_argument(
        "--word",
        type=int,
        default=2,
        help="word size: 1(INT8), 2(FP16)",
    )
    parser.add_argument(
        "--lin",
        type=int,
        default=2048,
        help="input sequence length",
    )
    parser.add_argument(
        "--lout",
        type=int,
        default=128,
        help="number of generated tokens",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="batch size",
    )

    parser.add_argument(
        "--sparsity",
        action="store_true",
        help="Enable sparsity-aware trace generation",
    )
    parser.add_argument(
        "--kv_budget",
        type=int,
        default=0,
        help="Fixed KV budget (used when >0)",
    )
    parser.add_argument(
        "--row_size",
        type=int,
        default=16,
        help="Tokens per row (default 16)",
    )
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.25,
        help="Sparsity ratio in ratio mode (0~1)",
    )
    parser.add_argument(
        "--kv_budget_dict",
        type=str,
        default="",
        help="Comma-separated budget list",
    )
    parser.add_argument(
        "--kv_budget_table",
        type=str,
        default="",
        help="KV budget table file path (.txt/.csv/.xlsx/.xls)",
    )

    args = parser.parse_args()

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    if args.gpu == "H100":
        gpu_device = GPUType.H100
    elif args.gpu == "A100a":
        gpu_device = GPUType.A100a
    else:
        assert 0

    if args.system == "dgx-attacc":
        print(
            "{}: ({} x {}), PIM:{}, [Lin, Lout, batch]: {}".format(
                args.system,
                args.gpu,
                args.ngpu,
                args.pim,
                [args.lin, args.lout, args.batch],
            )
        )
    else:
        print(
            "{}: ({} x {}), [Lin, Lout, batch]: {}".format(
                args.system, args.gpu, args.ngpu, [args.lin, args.lout, args.batch]
            )
        )

    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024

    output_path = "output.csv"
    if os.path.exists(output_path):
        os.remove(output_path)

    if args.kv_budget_dict:
        budget_list = [int(x) for x in args.kv_budget_dict.split(",")]
    else:
        budget_list = []

    # kv_budget_table overrides kv_budget_dict when provided.
    if args.kv_budget_table:
        if os.path.exists(args.kv_budget_table):
            budget_from_table = read_kv_budget_from_file(args.kv_budget_table)
            if budget_from_table:
                budget_list = budget_from_table
        else:
            exit(1)
    elif args.kv_budget_dict:
        budget_list = [int(x) for x in args.kv_budget_dict.split(",")]
    else:
        budget_list = []

    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)

    system = System(xpu_config["GPU"], modelinfos)

    # Expose sparsity knobs to System (read by System.set_accelerator -> Ramulator).
    system.sparsity = args.sparsity
    system.row_size = args.row_size
    system.sparsity_ratio = args.sparsity_ratio

    if budget_list:
        if len(budget_list) < args.lout:
            while len(budget_list) < args.lout:
                budget_list.append(budget_list[-1])

        kv_budget_dict = {args.lin + i + 1: b for i, b in enumerate(budget_list)}
        system.kv_budget_dict = kv_budget_dict
        system.kv_budget = budget_list[-1]
    else:
        system.kv_budget_dict = {}
        system.kv_budget = args.kv_budget

    if args.system in ["dgx-attacc"]:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA

        pim_config = make_pim_config(
            pim_type, InterfaceType.NVLINK3, power_constraint=args.powerlimit
        )
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config)

    elif args.system in ["dgx-cpu"]:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config["GPU"])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config["CPU"])


    run(
        system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit,
    )


if __name__ == "__main__":
    main()
