set -x

shopt -s expand_aliases

alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'

alias bk="/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=gfx942"
alias bt="/opt/rocm/bin/hipcc --offload-arch=gfx942  -g mla.cpp -o mla.exe"


# export HIP_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=3
# export HIP_VISIBLE_DEVICES=4
# export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7


# export LD_LIBRARY_PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
# export PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$PATH

rocm-smi | egrep "$HIP_VISIBLE_DEVICES    |Device"




function copy_recent_amdgcn_files() {
    # dir_name=paged_attn_decode
    # dir_name=paged_attn_decode_opt
    # dir_name=matmul_3d_kernel
    # dir_name=gemm_qk
    # dir_name=gemm_qk2
    # dir_name=gemm_qk_v2
    # dir_name=pa_decode_gluon
    # dir_name=sparse_attn_fwd_kernel
    # dir_name=block_sparse_attn
    # dir_name=block_sparse_attn_lxoptv1
    # dir_name=pa_decode_v2_big_blk_fp8
    # dir_name=pa_decode_v2_fp8_rtn
    # dir_name=pa_decode_v2_fp8
    dir_name=pa_decode_v2_gluon_fp8
    # local k=2
    local k=200
    # local dest_dir=$PWD/thread_trace/triton_gen_asm
    local dest_dir=$PWD/thread_trace/triton_gen_asm/$dir_name
    # rm -rf $dest_dir
    mkdir -p $dest_dir

    # kernel_name=paged_attn_decode
    # kernel_name=paged_attn_decode_opt
    # kernel_name=matmul_3d_kernel
    # kernel_name=gemm_qk
    # kernel_name=gemm_qk_v2
    # kernel_name=_paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk_gluon
    # kernel_name=block_sparse_attn
    # kernel_name=sparse_attn
    # kernel_name=pa_decode_v2_big_blk_fp8
    # kernel_name=pa_decode_v2_fp8
    # kernel_name=pa_decode_v2_gluon_fp8
    kernel_name=pa_decode_v2_gluon_big_blk_fp8

    file_filter="*$kernel_name*"

    ll ~/.triton/cache/*/$file_filter
    # cp ~/.triton/cache/*/$file_filter $dest_dir
    # ll $dest_dir

    # if [[ -z "$k" || -z "$dest_dir" ]]; then
    #     echo "Usage: copy_recent_amdgcn_files <number_of_files> <destination_directory>"
    #     return 1
    # fi

    # if [[ ! -d "$dest_dir" ]]; then
    #     echo "Error: Destination directory does not exist: $dest_dir"
    #     return 1
    # fi

    # local files=()
    # # while IFS= read -r line; do
    # #     files+=("$line")
    # # done < <(find ~/.triton/cache -type f -name $file_filter -print0 | xargs -0 ls -t | head -n "$k")
    # mapfile -t files < <(find ~/.triton/cache -type f -name $file_filter -print0 | xargs -0 ls -t | head -n "$k")
    # # echo $files

    # if [[ ${#files[@]} -eq 0 ]]; then
    #     echo "No .amdgcn files found in ~/.triton/cache"
    #     return 0
    # fi

    # echo "Copying ${#files[@]} most recent .amdgcn files to $dest_dir:"
    # # printf '%s\n' "${files[@]}"
    # for file in "${files[@]}"; do
    #     ll $file
    #     cp "$file" "$dest_dir/"
    # done

    amdgcn_filter=~/.triton/cache/*/*$kernel_name*.amdgcn
    cat $amdgcn_filter | egrep ".sgpr_count|.sgpr_spill_count|.vgpr_count|.vgpr_spill_count"
}


function run_triton_op {
    rm -rf ~/.triton/cache
    export AITER_LOG_MORE=1
    # unset AITER_LOG_MORE
    export TRITON_PRINT_AUTOTUNING=1
    # export TRITON_ALWAYS_COMPILE=1


    # ll ~/.triton/cache/*/*.hsaco
    # /opt/rocm/llvm/bin/clang++ -x assembler -mcode-object-version=5 -target amdgcn--amdhsa --offload-arch=gfx942 ./thread_trace/triton_gen_asm/block_sparse_attn/_triton_block_sparse_attn_fwd_kernel_v1.amdgcn -o /root/.triton/cache/VQ54VL322OSXBLRXJKYCSUKWTLMGD5SBDGBJUZMKHHO6CCS6PB5A/_triton_block_sparse_attn_fwd_kernel_v1.hsaco
    # /opt/rocm/llvm/bin/clang++ -x assembler -mcode-object-version=5 -target amdgcn--amdhsa --offload-arch=gfx942 ./thread_trace/triton_gen_asm/block_sparse_attn_lxoptv1/_triton_block_sparse_attn_fwd_kernel_v1.amdgcn -o /root/.triton/cache/FMGCXD6JK7RU6OEELBZHGQQNHHL4PSNEKXEVPM4KH7LU6KLILLIQ/_triton_block_sparse_attn_fwd_kernel_v1.hsaco
    # /opt/rocm/llvm/bin/clang++ -x assembler -mcode-object-version=5 -target amdgcn--amdhsa --offload-arch=gfx942 ./thread_trace/triton_gen_asm/block_sparse_attn_lxoptv1/_triton_block_sparse_attn_fwd_kernel_v1.amdgcn -o /root/.triton/cache/NCJ3W4CPTKSU75QT2NU5PBJRR2K3XRSAVBCJCHHEJVRDRQZTLU2A/_triton_block_sparse_attn_fwd_kernel_v1.hsaco
    # ll ~/.triton/cache/*/*.hsaco
    # md5sum ~/.triton/cache/*/*.hsaco

    # python ./00-gemm.py
    # python ./test_gluon_attn.py
    # python ./test_gluon_bmm.py
    # python ./test_gluon_pa_qk_v1.py
    # python ./test_gluon_pa_qk_v2.py
    # python ./test_pa_decode_gluon_triton.py

    # python ./test_pa_mtp.py -n 10,1 -q 2 -c 7 -b 32
    # python ./test_pa_mtp.py -n 8,2 -q 2 -c 57 -b 128 --block_size 1024 --trans_v

    # python ./test_pa_mtp.py -q 1 --block_size 16
    # python ./test_pa_mtp.py -q 1 --block_size 16 --trans_v
    # python ./test_pa_mtp.py -q 2 --block_size 16
    # python ./test_pa_mtp.py -q 2 --block_size 16 --trans_v

    # python ./test_pa_mtp.py -q 1 --block_size 1024
    # python ./test_pa_mtp.py -q 1 --block_size 1024 --trans_v
    # python ./test_pa_mtp.py -q 2 --block_size 1024
    # python ./test_pa_mtp.py -q 2 --block_size 1024 --trans_v


    # python ./test_pa_mtp.py -n 10,1 -c 4097 -b 32
    # python ./test_pa_mtp.py -n 10,1 -c 4096 -b 32
    # python ./test_pa_mtp.py -n 8,1 -c 4097 -b 32
    # python ./test_pa_mtp.py -n 8,1 -c 4096 -b 32
    # python ./test_pa_mtp.py -n 10,1 -c 4097 -b 32 --trans_v
    # python ./test_pa_mtp.py -n 10,1 -c 4096 -b 32 --trans_v
    # python ./test_pa_mtp.py -n 8,1 -c 4097 -b 32 --trans_v
    # python ./test_pa_mtp.py -n 8,1 -c 4096 -b 32 --trans_v


    # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 80 --block_size 16
    # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 128 --block_size 16
    python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 16
    # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4097 -b 128 --block_size 16

    # # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 80 --block_size 16 --trans_v
    # # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 128 --block_size 16 --trans_v
    # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 16 --trans_v

    # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 1024
    # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 1024 --trans_v
    # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4097 -b 128 --block_size 1024

    # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 64
    # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 64 --trans_v

    # python ./test_pa_mtp.py -n 8,1 -q 2 -c 4096 -b 128 --block_size 16
    # python ./test_pa_mtp.py -n 16,1 -q 2 -c 4096 -b 128 --block_size 16

    # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 32 --block_size 16
    # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 32 --block_size 1024
    # python ./test_pa_mtp.py -n 8,1 -q 2 -c 4096 -b 32 --block_size 16
    # python ./test_pa_mtp.py -n 16,1 -q 2 -c 4096 -b 32 --block_size 16

    # python ./test_pa_mtp.py -n 8,1 -q 1 -c 8192 -b 128 --block_size 16
    # python ./test_pa_mtp.py -n 8,1 -q 1 -b 128 --block_size 16
    # python ./test_pa_mtp.py -n 64,1 -q 1 --block_size 16

    # python ./test_pa_mtp.py -n 10,1 -q 1 -c 7 -b 32
    # python ./test_pa_mtp.py -n 10,1 -q 1 -c 256 -b 32
    # python ./test_pa_mtp.py -n 10,1 -q 1 -c 257 -b 32
    # python ./test_pa_mtp.py -n 10,1 -q 1 -c 512 -b 32
    # python ./test_pa_mtp.py -n 10,1 -q 1 -c 513 -b 32

    # rocprofv3 -i ./counters.yaml --kernel-include-regex "pa_decode_v2_fp8" --output-directory ./rocprofv3_out -- python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 32 --block_size 16
    # rocprofv3 -i ./counters.yaml --kernel-include-regex "_paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk" --output-directory ./rocprofv3_out -- python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 32 --block_size 16
    # rocprofv3 -i ./counters.yaml --kernel-include-regex "paged_attention_ll4mi_QKV_mfma16_kernel" --output-directory ./rocprofv3_out -- python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 32 --block_size 16

    # ll ~/.triton/cache/*/*.hsaco
    # md5sum ~/.triton/cache/*/*.hsaco

    copy_recent_amdgcn_files
}


function run_aiter_op {
    export AITER_LOG_MORE=1
    aiter_root_dir=/mnt/raid0/heyanguang/code/fa_triton/aiter
    pushd $aiter_root_dir


    # python3 op_tests/test_pa_mtp.py -n 10,1 -q 1
    python3 op_tests/test_pa_mtp.py -n 10,1 -q 3


    popd
}


function get_triton_pa_thread_trace {
    rm -rf ~/.triton/cache
    pushd $PWD
    export AITER_LOG_MORE=1

    # KERNEL=_fwd_kernel
    # KERNEL=_attn_fwd
    # KERNEL=_fwd_kernel_stage2
    # KERNEL=matmul_ori_kernel
    # KERNEL=matmul_ori_kernel_v2
    # KERNEL=_triton_mixed_sparse_attn_fwd_kernel_v1
    # KERNEL=_triton_block_sparse_attn_fwd_kernel_v1
    # KERNEL=pa_decode_v2_gluon_fp8
    KERNEL=pa_decode_v2_gluon_big_blk_fp8
    # KERNEL=pa_bf16_pertokenFp8_gqa8_2tg_4w_uhp
    # KERNEL=matmul_kernel
    # KERNEL=_fwd_grouped_kernel_stage1_rope
    # export KERNEL_VERSION="triton_pa_prefill_bf16"
    # export KERNEL_VERSION="triton_mha_fwd_bf16"
    # export KERNEL_VERSION="triton_${KERNEL}_bf16"
    # export KERNEL_VERSION="triton_${KERNEL}_bf16_v2"

    export KERNEL_VERSION="${KERNEL}_v1"
    # export KERNEL_VERSION="${KERNEL}_v1_rm_v_scale"

    # pytest ./test_pa_prefill.py::test_contexted_kv_attention -v -s -k "0-cuda:0-auto-dtype1-128-1-4"
    # pytest ./test_pa_prefill.py::test_mha -v -s -k "False-True-0.0-False-False-128-4-4-1024-2048-2"
    # pytest ./test_pa_prefill.py::test_mha -v -s -k "True-False-0.0-False-False-128-16-1-1024-1024-80"

    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py --model all --seqlen 16384
    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 320 --model all --seqlen 8192
    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens -print_vgpr
    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens


    DUMP_TRACE=1
    # DUMP_TRACE=0
    if [ $DUMP_TRACE = 1 ]; then
        trace_dir=./thread_trace/trace_${KERNEL_VERSION}
        rm -rf ./${trace_dir} ./${trace_dir}.tar.gz
        mkdir -p ${trace_dir}

        rocprofv2 -d ${trace_dir} -i ./thread_trace/att.txt --plugin att auto --mode file,csv -o ${trace_dir}/csv_${KERNEL_VERSION} \
        python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 1024
        # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 16
        # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 80 --block_size 16
        # python ./block_sparse_attn.py
        # python ./mixed_sparse_attn.py
        # python ./00-gemm.py
        # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens
        # pytest ./test_pa_prefill.py::test_mha -v -s -k "False-True-0.0-False-False-128-4-4-1024-2048-2"
        # pytest ./test_pa_prefill.py::test_contexted_kv_attention -v -s -k "0-cuda:0-auto-dtype1-128-1-4"

        cd ./thread_trace
        tar -zcf ./trace_${KERNEL_VERSION}.tar.gz ./trace_${KERNEL_VERSION}
        ls -lah ./trace_${KERNEL_VERSION} ./trace_${KERNEL_VERSION}.tar.gz
        cd -
    fi

    copy_recent_amdgcn_files
    popd
}





# pip install -e .

# install aiter
# python3 setup.py develop

# install triton
# pip install -e python
# pip install -e .


# for i in {1..1..1}; do
# # for i in {1..5..1}; do
# # for i in {1..20..1}; do
#     echo "*******************************iter=$i*******************************"

# done


run_triton_op
# run_aiter_op
# get_triton_pa_thread_trace



# KERNEL=kernel
# ll ~/.triton/cache/*/*$KERNEL*.amdgcn

# rocprofv2 --version
# cat /etc/os-release


set +x
