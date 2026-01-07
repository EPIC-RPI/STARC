## initialize ramulator2 
cd ramulator2
git reset --hard b7c70275f04126c647edb989270cc429776955d1

# Portability fix: ensure uint64_t is defined on all toolchains
# (This commit relies on transitive includes on some systems.)
UTILS_H="src/base/utils.h"
if ! grep -qE '^\s*#include\s*<cstdint>\s*$' "$UTILS_H"; then
  # Insert after the header guard or the first includes block.
  # Safe/Idempotent: only adds the include once.
  sed -i '1{/^#pragma once$/a#include <cstdint>\n}' "$UTILS_H"
  # If the file doesn't start with #pragma once, insert at very top.
  if ! head -n 3 "$UTILS_H" | grep -q "<cstdint>"; then
    sed -i '1i#include <cstdint>\n' "$UTILS_H"
  fi
fi

cd ..

# copy files
cp pim_ramulator_src/attacc_bank.yaml ramulator2/
cp pim_ramulator_src/attacc_bg.yaml ramulator2/
cp pim_ramulator_src/attacc_buffer.yaml ramulator2/
cp pim_ramulator_src/HBM3_base.yaml ramulator2/
cp pim_ramulator_src/hbm3_linear_mappers.cpp ramulator2/src/addr_mapper/impl/
cp pim_ramulator_src/hbm3_pim_linear_mappers.cpp ramulator2/src/addr_mapper/impl/
cp pim_ramulator_src/HBM3-PIM.cpp ramulator2/src/dram/impl/
cp pim_ramulator_src/hbm3_controller.cpp ramulator2/src/dram_controller/impl/
cp pim_ramulator_src/hbm3_pim_controller.cpp ramulator2/src/dram_controller/impl/
cp pim_ramulator_src/hbm3_trace_recorder.cpp ramulator2/src/dram_controller/impl/plugin/
cp pim_ramulator_src/all_bank_refresh_hbm3.cpp ramulator2/src/dram_controller/impl/refresh/
cp pim_ramulator_src/no_refresh.cpp ramulator2/src/dram_controller/impl/refresh/
cp pim_ramulator_src/pim_scheduler.cpp ramulator2/src/dram_controller/impl/scheduler/
cp pim_ramulator_src/pim_loadstore_trace.cpp ramulator2/src/frontend/impl/memory_trace/
cp pim_ramulator_src/PIM_DRAM_system.cpp ramulator2/src/memory_system/impl/
cp -r pim_ramulator_src/trace_gen ramulator2/
cp -r pim_ramulator_src/patches ramulator2/

# Apply patches
cd ramulator2;

for f in ./patches/*.patch
do
    patch -p1 < $f
done
