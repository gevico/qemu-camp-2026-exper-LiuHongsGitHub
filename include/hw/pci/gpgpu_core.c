/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"
// #include "exec/memory.h"
// #include "exec/cpu_ldst.h"

/* TODO: Implement warp initialization */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    (void)pc;
    (void)thread_id_base;
    (void)block_id;
    (void)num_threads;
    (void)warp_id;
    (void)block_id_linear;
    //清零 warp结构
    memset(warp, 0, sizeof(*warp));
    //init warp meta data
    warp->active_mask = num_threads >= 32 ? 0xFFFFFFFF : (1<< num_threads)-1;
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    memcpy(warp->block_id, block_id, sizeof(warp->block_id ));
    for (int i =0;i<GPGPU_WARP_SIZE;i++){
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear,warp_id,i);
        lane->fcsr = 0;
        lane->active = i < num_threads ;
        if (lane->active) {
            lane->gpr[1] = thread_id_base + i;
        }
        set_default_nan_mode(1,&lane->fp_status);
    }
   
}

static inline uint32_t sext32(uint32_t val,int bit_len){
    int32_t m = 1U << (bit_len - 1);
    return (val ^ m)-m;
}


static inline void decode_and_exec(GPGPUState *s,GPGPULane *lane, uint32_t inst) {
    uint8_t opcode = inst & 0x7F;
    uint8_t rd = (inst >> 7) & 0x1F;
    uint8_t rs1 = (inst >> 15) & 0x1F;
    uint8_t rs2 = (inst >> 20) & 0x1F;

    uint32_t imm_i = sext32(inst >> 20, 12);
    uint32_t imm_s = sext32(((inst >> 25) << 5) | ((inst >> 7) & 0x1F), 12);
    uint32_t imm_b = sext32(((inst >> 31) << 12) | ((inst >> 7) & 1) << 11 |
                           ((inst >> 25) & 0x3F) << 5 | ((inst >> 8) & 0xF) << 1, 13);
    uint32_t imm_u = inst & 0xFFFFF000;
    int32_t imm_j = sext32(((inst >> 31) << 20) | ((inst >> 12) & 0xFF) << 12 |
                           ((inst >> 20) & 1) << 11 | ((inst >> 21) & 0x3FF) << 1, 21);
    uint32_t shamt = (inst >> 20) & 0x1F;
    uint32_t funct3 = (inst >> 12) & 0x7;
    uint32_t funct7 = (inst >> 25) & 0x7F;

    uint8_t rm = (inst >> 12) & 0x7;
    uint8_t rs3 = (inst >> 27) & 0x1F;
    switch(opcode){
        case 0x37:
            if (rd != 0) lane->gpr[rd] = imm_u;
            return;
        
        case 0x17:
            if (rd != 0) lane->gpr[rd] = lane->pc + imm_u -4;
            return;
        case 0x6F:
            if (rd != 0) lane->gpr[rd] = lane->pc;
            lane->pc = lane->pc + imm_j - 4;
            return;

        /* JALR: Jump and Link Register */
        case 0x67:
            {
                uint32_t tmp = lane->pc;
                lane->pc = (lane->gpr[rs1] + imm_i) & ~1;
                if (rd != 0) lane->gpr[rd] = tmp;
            }
            return;

        /* BRANCH: Bxx */
        case 0x63:
            {
                bool taken = false;
                int32_t rs1_val = (int32_t)lane->gpr[rs1];
                int32_t rs2_val = (int32_t)lane->gpr[rs2];
                switch (funct3) {
                case 0: taken = (rs1_val == rs2_val); break;       /* BEQ */
                case 1: taken = (rs1_val != rs2_val); break;       /* BNE */
                case 4: taken = (rs1_val < rs2_val); break;        /* BLT */
                case 5: taken = (rs1_val >= rs2_val); break;       /* BGE */
                case 6: taken = (lane->gpr[rs1] < lane->gpr[rs2]); break;  /* BLTU */
                case 7: taken = (lane->gpr[rs1] >= lane->gpr[rs2]); break; /* BGEU */
                default: return;
                }
                if (taken) lane->pc = lane->pc + imm_b - 4;
            }
            return;

        /* LOAD: LB, LH, LW, LBU, LHU */
        case 0x03:
            {
                uint32_t addr = lane->gpr[rs1] + imm_i;
                uint32_t val = address_space_ldl_le(pci_get_address_space(PCI_DEVICE(s)),
                                                    addr, MEMTXATTRS_UNSPECIFIED, NULL);
                switch (funct3) {
                case 0: val = sext32(val & 0xFF, 8); break;       /* LB */
                case 1: val = sext32(val & 0xFFFF, 16); break;     /* LH */
                case 2: break;                                       /* LW */
                case 4: val = val & 0xFF; break;                    /* LBU */
                case 5: val = val & 0xFFFF; break;                 /* LHU */
                default: return;
                }
                if (rd != 0) lane->gpr[rd] = val;
            }
            return;

        /* STORE: SB, SH, SW */
        case 0x23:
            {
                uint32_t addr = lane->gpr[rs1] + imm_s;
                uint32_t val = lane->gpr[rs2];
                switch (funct3) {
                    case 0: 
                        address_space_stb_le(pci_get_address_space(PCI_DEVICE(s)),
                            addr, val & 0xFF, MEMTXATTRS_UNSPECIFIED, NULL);
                        break;
                    case 1:
                        address_space_stw_le(pci_get_address_space(PCI_DEVICE(s)),
                            addr, val & 0xFFFF, MEMTXATTRS_UNSPECIFIED, NULL);
                        break;
                    case 2: 
                        address_space_stl_le(pci_get_address_space(PCI_DEVICE(s)),
                                    addr, val, MEMTXATTRS_UNSPECIFIED, NULL);
                        break;
                    default: return;
                }
            }
            return;

        /* OP-IMM: Integer Register-Immediate Operations */
        case 0x13:
            {
                int32_t rs1_val = (int32_t)lane->gpr[rs1];
                uint32_t result;
                switch (funct3) {
                case 0: result = rs1_val + (int32_t)imm_i; break;         /* ADDI */
                case 1: result = lane->gpr[rs1] << shamt; break;          /* SLLI */
                case 2: result = (rs1_val < (int32_t)imm_i) ? 1 : 0; break;  /* SLTI */
                case 3: result = (lane->gpr[rs1] < (uint32_t)imm_i) ? 1 : 0; break; /* SLTIU */
                case 4: result = lane->gpr[rs1] ^ imm_i; break;           /* XORI */
                case 5:
                    if (funct7 == 0) result = lane->gpr[rs1] >> shamt;    /* SRLI */
                    else result = ((int32_t)lane->gpr[rs1]) >> shamt;    /* SRAI */
                    break;
                case 6: result = lane->gpr[rs1] | imm_i; break;          /* ORI */
                case 7: result = lane->gpr[rs1] & imm_i; break;          /* ANDI */
                default: return;
                }
                if (rd != 0) lane->gpr[rd] = result;
            }
            return;

        /* OP: Integer Register-Register Operations */
        case 0x33:
            {
                int32_t rs1_val = (int32_t)lane->gpr[rs1];
                int32_t rs2_val = (int32_t)lane->gpr[rs2];
                int32_t result;
                switch (funct3) {
                case 0:
                    result = (funct7 == 0x20) ? (rs1_val - rs2_val) : (rs1_val + rs2_val);
                    break;  /* ADD/SUB */
                case 1: result = lane->gpr[rs1] << (lane->gpr[rs2] & 0x1F); break;  /* SLL */
                case 2: result = (rs1_val < rs2_val) ? 1 : 0; break;                /* SLT */
                case 3: result = (lane->gpr[rs1] < lane->gpr[rs2]) ? 1 : 0; break;  /* SLTU */
                case 4: result = lane->gpr[rs1] ^ lane->gpr[rs2]; break;             /* XOR */
                case 5:
                    if (funct7 == 0) result = lane->gpr[rs1] >> (lane->gpr[rs2] & 0x1F);
                    else result = rs1_val >> (lane->gpr[rs2] & 0x1F);
                    break;  /* SRL/SRA */
                case 6: result = lane->gpr[rs1] | lane->gpr[rs2]; break;              /* OR */
                case 7: result = lane->gpr[rs1] & lane->gpr[rs2]; break;              /* AND */
                default: return;
                }
                if (rd != 0) lane->gpr[rd] = result;
            }
            return;

        /* FENCE (nop) */
        case 0x0F:
            return;

        /* ECALL/EBREAK */
        case 0x73:
            /* Exit condition: x17 == 1 means "exit(0)" */
            if (rd == 0 && rs1 == 0 && funct3 == 0) {
                /* EBREAK: mark thread for exit */
                lane->gpr[17] = 1;
            }
            return;
        case 0x07: /* FLV / FLW */  
            {
                if (funct3 == 0x2) {
                    uint32_t addr = lane->gpr[rs1] + imm_i; 
                    uint32_t val = address_space_ldl_le(
                        pci_get_address_space(PCI_DEVICE(s)), addr, MEMTXATTRS_UNSPECIFIED, NULL);
                    if (rd != 0) lane->fpr[rd] = val;
                }
            }
            return;
        case 0x27:
            {
             if (funct3 == 0x2) {
                    uint32_t addr = lane->gpr[rs1] + imm_s; 
                    uint32_t val = lane->fpr[rs2];
                    address_space_stl_le(pci_get_address_space(PCI_DEVICE(s)),
                                    addr, val, MEMTXATTRS_UNSPECIFIED, NULL);
                }
            }
            return;
        case 0x53: /* OP-FP */
            switch (funct7) {
            case 0x00: /* FADD.S */  
                
            case 0x04: /* FSUB.S */    
            case 0x08: /* FMUL.S */    ...
            case 0x0C: /* FDIV.S */    ...
            /* ... 更多浮点指令 ... */
            }
            break;
        default:
            /* Unknown opcode - could set error */
            return;
    }
}

/* TODO: Implement warp execution (RV32I + RV32F interpreter) */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    for(uint32_t cycles = 0; cycles < max_cycles; cycles++){
        if (warp->active_mask == 0){
            return 0;
        }
        for(int i =0 ; i< GPGPU_WARP_SIZE;i++){
            GPGPULane *lane = &warp->lanes[i];
            if (!lane->active){
                continue;
            }
            uint32_t inst ;
            if(lane->pc < s->vram_size){
                inst = *(uint32_t*)(s->vram_ptr+lane->pc);
            }else if (lane->pc >= GPGPU_CORE_CTRL_BASE){
                 inst = read_ctrl_reg(s, lane->pc);
            }else {
                inst = address_space_ldl_le(
                    pci_get_address_space(PCI_DEVICE(s)),
                    lane->pc, MEMTXATTRS_UNSPECIFIED, NULL);
            }
            lane->pc += 4;
            //general reg a7 == 1 mean syscall exit
            decode_and_exec(s,lane,inst);
            if (lane->gpr[17] == 1){
                lane->active = false;
                warp->active_mask &= ~(1U<<i);
            }
        }
    }
    
 
    return 0;
}

/* TODO: Implement kernel dispatch and execution */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t grid_dim_x = s->kernel.grid_dim[0];
    uint32_t grid_dim_y = s->kernel.grid_dim[1];
    uint32_t grid_dim_z = s->kernel.grid_dim[2];
    uint32_t block_dim_x = s->kernel.block_dim[0];
    uint32_t block_dim_y = s->kernel.block_dim[1];
    uint32_t block_dim_z = s->kernel.block_dim[2];
    
    uint32_t threads_per_block = block_dim_x * block_dim_y * block_dim_z;
    uint32_t warps_per_block = (threads_per_block + 31) / 32;
    uint64_t kernel_addr = s->kernel.kernel_addr;
    
    GPGPUWarp *warps = g_malloc(sizeof(GPGPUWarp) * warps_per_block);
    
    // 遍历所有 block
    for (uint32_t bz = 0; bz < grid_dim_z; bz++) {
        for (uint32_t by = 0; by < grid_dim_y; by++) {
            for (uint32_t bx = 0; bx < grid_dim_x; bx++) {
                uint32_t block_id[3] = {bx, by, bz};
                uint32_t block_linear = bz * grid_dim_x * grid_dim_y 
                                      + by * grid_dim_x + bx;
                
                // 初始化 block 内所有 warp
                for (uint32_t w = 0; w < warps_per_block; w++) {
                    uint32_t tid_base = block_linear * threads_per_block 
                                       + w * 32;
                    gpgpu_core_init_warp(&warps[w], kernel_addr, 
                                         tid_base, block_id,
                                         threads_per_block - w * 32,
                                         w, block_linear);
                }
                
                // 执行 block 直到所有 warp 完成
                while (1) {
                    bool all_done = true;
                    for (uint32_t w = 0; w < warps_per_block; w++) {
                        if (warps[w].active_mask != 0) {
                            all_done = false;
                            gpgpu_core_exec_warp(s, &warps[w], 1000);
                        }
                    }
                    if (all_done) break;
                }
            }
        }
    }
    
    g_free(warps);
    return 0;
}

