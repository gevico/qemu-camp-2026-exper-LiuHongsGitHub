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
#include "system/memory.h"
#include "hw/pci/pci.h"
// #include "exec/cpu_ldst.h"

/* TODO: Implement warp initialization */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    //清零 warp结构
    memset(warp, 0, sizeof(*warp));
    //init warp meta data
    warp->active_mask = num_threads == 32 ? 0xFFFFFFFF : (1<< num_threads)-1;
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    memcpy(warp->block_id, block_id, sizeof(warp->block_id ));
    for (int i =0;i<GPGPU_WARP_SIZE;i++){
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear,warp_id,i);
        lane->fcsr = 0;
        lane->active = i < num_threads ;
        if (lane -> active) {
            lane->gpr[1] = thread_id_base + i;
        }
        set_default_nan_mode(1,&lane->fp_status);
    }
   
}

static uint32_t sext32(uint32_t val,int bit_len){
    int32_t m = 1U << (bit_len - 1);
    return (val ^ m)-m;
}
static inline uint8_t get_rm(GPGPULane *lane){
    return (lane->fcsr >> 5) & 0x7;
}
static inline void update_fcsr(GPGPULane *lane){
    uint32_t flags = lane->fp_status.float_exception_flags;
    lane->fcsr |= flags & 0x1F;
}
static inline FloatRoundMode rv_rm_to_soft(uint8_t rm){
    static const FloatRoundMode map[] = {
        float_round_nearest_even,
        float_round_to_zero,
        float_round_up,
        float_round_down,
        float_round_to_zero,
    };
    if (rm<=4){
        return map[rm];
    }
     /* For unsupported rounding modes, default to round to zero */
     
    
    return float_round_nearest_even;
}


static inline void decode_and_exec(GPGPUState *s,GPGPULane *lane, uint32_t inst) {
    uint8_t opcode = inst & 0x7F;
    uint8_t rd = (inst >> 7) & 0x1F;
    uint8_t rs1 = (inst >> 15) & 0x1F;
    uint8_t rs2 = (inst >> 20) & 0x1F;
    uint8_t rs3 = (inst >> 27) & 0x1F;

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
                uint32_t val = 0;
                switch (funct3) {
                case 0: /* LB */
                    uint8_t b = address_space_ldub(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                                    addr, MEMTXATTRS_UNSPECIFIED, NULL);
                    val = sext32(b, 8);
                    break;       
                case 1: /* LH */
                    {
                    uint16_t h = address_space_lduw_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                                    addr, MEMTXATTRS_UNSPECIFIED, NULL);
                    val = sext32(h, 16); 
                    }
                    break;     
                case 2: /* LW */
                    val = address_space_ldl_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                                    addr, MEMTXATTRS_UNSPECIFIED, NULL);
                    break;                                       
                case 4: /* LBU */
                    val = address_space_lduw_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                                    addr, MEMTXATTRS_UNSPECIFIED, NULL);
                    break;                    
                case 5: /* LHU */
                    {
                    uint16_t h = address_space_lduw_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                                    addr, MEMTXATTRS_UNSPECIFIED, NULL);
                    val = (uint32_t)h;
                    }
                    
                    break;                 
                default: return;
                }
                if (rd != 0) lane->gpr[rd] = val;
            }
            return;

        /* STORE: SB, SH, SW */
        case 0x23:
            {   
                if (funct3 == 0){
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    uint32_t val = lane->gpr[rs2] & 0xFF;
                    address_space_stb(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                        addr, val, MEMTXATTRS_UNSPECIFIED, NULL);
                } else if (funct3 == 1){
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    uint32_t val = lane->gpr[rs2] & 0xFFFF;
                    address_space_stw_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                        addr, val, MEMTXATTRS_UNSPECIFIED, NULL);
                } else if (funct3 == 2){
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    address_space_stl_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                        addr, lane->gpr[rs2], MEMTXATTRS_UNSPECIFIED, NULL);
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
                    else result = (uint32_t)(rs1_val >> (lane->gpr[rs2] & 0x1F));
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
        case 0x07:
            {
                if (funct3 == 2 && rd != 0){
                    uint32_t addr = lane->gpr[rs1] +imm_i;
                    uint32_t val = address_space_ldl_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                                        addr, MEMTXATTRS_UNSPECIFIED, NULL);
                    lane->fpr[rd] = val;
                }
                return;
            }
        //funct3 = 0x2   
        case 0x27:
            {
                if (funct3 ==2 && rs2 != 0){
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    address_space_stl_le(pci_device_iommu_address_space(PCI_DEVICE(s)),
                                        addr, lane->fpr[rs2], MEMTXATTRS_UNSPECIFIED, NULL);
                }
            }
            return;
        case 0x43:
            {
                float32 fs1_fmadd = make_float32(lane->fpr[rs1]);  // 声明局部变量
                float32 fs2_fmadd = make_float32(lane->fpr[rs2]);
                float32 fs3_fmadd = make_float32(lane->fpr[rs3]);
            
                float32 result_fmadd = float32_muladd(fs1_fmadd,fs2_fmadd,fs3_fmadd,0,&lane->fp_status);
                if (rd != 0) lane->fpr[rd] = (uint32_t)result_fmadd;
            }
            return;
        case 0x47:
            {
                float32 fs1_fmsub = make_float32(lane->fpr[rs1]);
                float32 fs2_fmsub = make_float32(lane->fpr[rs2]);
                float32 fs3_fmsub = make_float32(lane->fpr[rs3]);
                float32 result_fmsub = float32_muladd(fs1_fmsub,fs2_fmsub,fs3_fmsub,float_muladd_negate_c,&lane->fp_status);
                if (rd != 0) lane->fpr[rd] = (uint32_t)result_fmsub;
            }
            return;
        case 0x4B:
            {
                float32 fs1_fnmsub  = make_float32(lane->fpr[rs1]);
                float32 fs2_fnmsub  = make_float32(lane->fpr[rs2]);
                float32 fs3_fnmsub  = make_float32(lane->fpr[rs3]);
                float32 result_fnmsub = float32_muladd(fs1_fnmsub,fs2_fnmsub,fs3_fnmsub,float_muladd_negate_product,&lane->fp_status);
                if (rd != 0) lane->fpr[rd] = (uint32_t)result_fnmsub;
            }
            return;
        case 0x4F:/* FNMADD.S - Fused NegMultiply-Add: rd = -(fs1 * fs2) - fs3 */
            {
                float32 fs1_fnmadd = make_float32(lane->fpr[rs1]);
                float32 fs2_fnmadd = make_float32(lane->fpr[rs2]);
                float32 fs3_fnmadd = make_float32(lane->fpr[rs3]);
                float32 result_fnmadd = float32_muladd(fs1_fnmadd,fs2_fnmadd,fs3_fnmadd,float_muladd_negate_product | float_muladd_negate_c,&lane->fp_status);
                if (rd != 0) lane->fpr[rd] = (uint32_t)result_fnmadd;
            }
            return;
        case 0x53:
            {
                lane->fp_status.float_rounding_mode = rv_rm_to_soft(get_rm(lane));
                float32 fs1 = make_float32(lane->fpr[rs1]);
                float32 fs2 = make_float32(lane->fpr[rs2]);
                float32 result = make_float32(0);
                switch (funct7) {
                    case 0x00: 
                        if (funct3 == 0) {
                            result = float32_add(fs1,fs2,&lane->fp_status);
                            if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        } else if (funct3 == 1) {
                            result = float32_sub(fs1,fs2,&lane->fp_status);
                            if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        }
                        break;
                    case 0x08: 
                        if (funct3 == 0) {
                            result = float32_mul(fs1,fs2,&lane->fp_status);
                            if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        }
                        break;
                    case 0x18://FDIV.S
                        if (funct3 == 0) {
                            result = float32_div(fs1,fs2,&lane->fp_status);
                            if (rd != 0) {
                                lane->fpr[rd] = (uint32_t)result;
                            }
                        }
                        break;
                    case 0x20: 
                        if (funct3 == 0) {
                            result = make_float32((float32_val(fs1) & ~0x80000000) | (float32_val(fs2) & 0x80000000));
                        } else if (funct3 == 1) {
                            result = make_float32((float32_val(fs1) & ~0x80000000) | (~float32_val(fs2) & 0x80000000));
                        } else if (funct3 == 2) {
                            result = make_float32(float32_val(fs1) ^ (float32_val(fs2)&0x80000000));
                        }
                        if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        break;
                    case 0x28: 
                        if (funct3 == 0) {
                            result = float32_min(fs1,fs2,&lane->fp_status);
                            if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        }
                        break;
                    case 0x29:
                        if (funct3 == 0) {
                            result = float32_max(fs1,fs2,&lane->fp_status);
                            if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        }
                        break;
                    case 0x50:
                        switch (funct3){
                            case 0:
                                if (float32_le_quiet(fs1,fs2,&lane->fp_status)){
                                    if (rd !=0) lane->gpr[rd] = 1;
                                }else {
                                    if (rd !=0) lane->gpr[rd] = 0;
                                }
                                break;
                            case 1:
                                if (float32_lt_quiet(fs1,fs2,&lane->fp_status)){
                                    if (rd !=0) lane->gpr[rd] = 1;
                                }else {
                                    if (rd !=0) lane->gpr[rd] = 0;
                                }   
                                break;
                            case 2:
                                if (float32_eq_quiet(fs1,fs2,&lane->fp_status)){
                                    if (rd !=0) lane->gpr[rd] = 1;
                                }else {
                                    if (rd !=0) lane->gpr[rd] = 0;
                                }   
                                break;
                            default:
                                break;
                        }
                        break;
                    case 0x58:
                        if (funct3 == 0) {
                            result = float32_sqrt(fs1,&lane->fp_status);
                            if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        }
                        break;
                    case 0x60: 
                        if (funct3 == 0) {
                            lane->gpr[rd] = float32_to_int32(fs1,&lane->fp_status);
                        } else if (funct3 == 1) {
                            lane->fpr[rd] = (uint32_t)float32_to_uint32(fs1,&lane->fp_status);
                        }
                        break;
                    case 0x68: 
                        if (funct3 == 0){
                            result = int32_to_float32((int32_t)lane->gpr[rs1],&lane->fp_status);
                            lane->fpr[rd] = (uint32_t)result;
                        }
                        break;
                    case 0x70: 
                        if (funct3 == 1){// FCLASS.S → 返回10位掩码到整数寄存器
                            uint32_t fclass = 0;
                            uint32_t val = float32_val(fs1);
                            bool sign = (val >> 31) & 1;
                            uint32_t exp = (val >> 23) & 0xFF;
                            uint32_t mantissa = val & 0x7FFFFF;
                            if (exp == 0xFF && mantissa != 0) {
                                fclass |= 1 << (sign ? 0 : 7); // Signaling NaN
                            } else if (exp == 0xFF && mantissa == 0) {
                                fclass |= 1 << (sign ? 1 : 6); // Infinity
                            } else if (exp == 0 && mantissa == 0) {
                                fclass |= 1 << (sign ? 2 : 5); // Zero
                            } else if (exp == 0 && mantissa != 0) {
                                fclass |= 1 << (sign ? 3 : 4); // Subnormal
                            } else {
                                fclass |= 1 << (sign ? 8 : 9); // Normal
                            }
                            if (rd != 0) {
                                lane->gpr[rd] = fclass;
                            }
                        }
                        if (funct3 == 0){
                            lane->gpr[rd] = lane->fpr[rs1];
                        }
                        break;
                    case 0x69:
                        if (funct3 == 0){
                            result = uint32_to_float32(lane->gpr[rs1], &lane->fp_status);
                            if (rd != 0) lane->fpr[rd] = (uint32_t)result;
                        }
                        break;
                    case 0x78: 
                        if (funct3 == 0){
                            lane->fpr[rd] = lane->gpr[rs1];
                        }
                        break;
                    
                }
                update_fcsr(lane);
                return;
            }
        default:
            /* Unknown opcode - could set error */
            return;
    }
}

uint32_t read_ctrl_reg(GPGPUState *s, uint32_t addr) {
    uint32_t offset = addr - GPGPU_CORE_CTRL_BASE;
    switch (offset) {
        case 0x00:
            return s->simt.thread_id[0]; /* threadIdx.x */
        case 0x04:
            return s->simt.thread_id[1]; /* threadIdx.y */
        case 0x08:
            return s->simt.thread_id[2]; /* threadIdx.z */

        case 0x10:
            return s->simt.block_id[0]; /* blockIdx.x */
        case 0x14:
            return s->simt.block_id[1]; /* blockIdx.y */    
        case 0x18:
            return s->simt.block_id[2]; /* blockIdx.z */
        
        case 0x20:
            return s->kernel.block_dim[0]; /* warp_id */
        case 0x24:
            return s->kernel.block_dim[1]; /* lane_id */   
        case 0x28:
            return s->kernel.block_dim[2]; /* blockDim.x */

        case 0x30:
            return s->kernel.grid_dim[0]; /* gridDim.x */
        case 0x34:
            return s->kernel.grid_dim[1]; /* gridDim.y */
        case 0x38:  
            return s->kernel.grid_dim[2]; /* gridDim.z */
        default:
            /* Invalid register address - could set error */
            qemu_log_mask(LOG_GUEST_ERROR,"Invalid control register address: 0x%x\n", addr);
            return 0;
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
            if(lane->pc + 4 <= s->vram_size){
                //pc in vram scope -> get instruction from 
                inst = *(uint32_t*)(s->vram_ptr+lane->pc);
            }else if (lane->pc >= GPGPU_CORE_CTRL_BASE){
                //pc in ctrl area, read from virtual register
                inst = read_ctrl_reg(s, lane->pc);
            }else {
                //other address -> read system memory by PCI IOMMU  
                inst = address_space_ldl_le(
                    pci_device_iommu_address_space(PCI_DEVICE(s)),
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

