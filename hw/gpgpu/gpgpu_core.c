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
            uint32_t csr_imm = (inst >> 20) & 0x1F;
            if (funct3 == 0 && rd ==0 && rs1 == 0){
                lane->gpr[17] =1;
            } else if (funct3 == 2 || funct3 == 3 ||funct3 == 6 ||funct3 == 7) {
                uint32_t csr_val = 0;
                switch (csr_imm) {
                    case CSR_MHARTID:
                        csr_val = lane->mhartid;
                        break;
                    case CSR_FFLAGS:
                        csr_val = lane->fcsr & 0x1F;
                        break;
                    case CSR_FRM:
                        csr_val = (lane->fcsr >> 5) & 0x7;
                        break;
                    case CSR_FCSR:
                        csr_val = lane->fcsr;
                        break;
                }
                if (rd != 0) lane->gpr[rd] = csr_val;
            }
            /* Exit condition: x17 == 1 means "exit(0)" */
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
                    case 0x22:
                        switch (rs2) {
                            case 0:
                                uint16_t bf16_val = lane->fpr[rs1] & 0xFFFF;
                                uint32_t f32_val = ((uint32_t)bf16_val) << 16;
                                lane->fpr[rd] = f32_val;
                                break;
                            case 1:
                                uint32_t f32_val = lane->fpr[rs1];
                                uint16_t bf16_val = (f32_val >> 16) & 0xFFFF;
                                lane->fpr[rd] = bf16_val;
                                break;
                            default:
                                break;
                        }
                        update_fcsr(lane);
                        break;
                    case 0x24: /* FP8 E4M3/E5M2 */
                        switch (rs2) {
                            case 0:{/* fcvt.s.e4m3  fd, fs1  — E4M3 → FP32 (上行读) */
                                // 从 fs1 取出 E4M3 格式的 8-bit 值
                                // 转换为 FP32 格式
                                // 写入 lane->fpr[rd]
                                uint8_t e4m3_bits  = lane->fpr[rs1] & 0xFF;
                                 // 解码 E4M3: S(1) + E(4) + M(3)
                                bool sign = (e4m3_bits >> 7) & 1;
                                uint8_t exp4 = (e4m3_bits >> 3) & 0xF;
                                uint8_t mantissa3 = e4m3_bits & 0x7;
                                uint32_t f32_val;
                                if (exp4 == 0 && mantissa3 == 0) {
                                    // 0
                                    f32_val = sign ? 0x80000000 : 0x00000000;
                                } else if (exp4 == 0) {
                                    //subnormal: value = 0.mantissa3 * 2^(-6)
                                    //find the position of the highest 1 in mantissa3
                                    int msb = 2;
                                    while (msb >= 0 && !((mantissa3 >> msb) & 1)) {
                                        msb--;
                                    }
                                    int32_t true_exp = -6 - (2 - msb); // 需要右移的位数
                                    uint32_t norm_m = (mantissa3 << (21 - msb)) & 0x7FFFFF; // 将有效位左移到 FP32 的位置
                                    f32_val = ((uint32_t)sign << 31) | ((uint32_t)(true_exp + 127) << 23) | norm_m; // 加上 FP32 的指数偏
                                } else if (exp4 == 0xF) {
                                    // E4M3 不应有 Inf/NaN，按最大值处理
                                    // saturate to max: sign ? -448 : +448
                                    f32_val = sign ? 0xC3E00000 : 0x43E00000;
                                } else {
                                    // 正常数
                                    int32_t exp = exp4 - 7 + 127; // E4M3 的指数偏移是 7，FP32 的指数偏移是 127
                                    uint32_t mantissa = (mantissa3 << 20) & 0x7FFFFF; // 将 M 部分左移到 FP32 的位置
                                    f32_val = ((uint32_t)sign << 31) | (exp << 23) | mantissa | 0x40000000;  // 加隐含的 1
                                }
                                if (rd != 0) lane->fpr[rd] = f32_val;
                            }
                            break;
                            case 1:{//fcvt.e4m3.s fd,fs1 - FP32 → E4M3 (下行写)
                                uint32_t bits = lane->fpr[rs1];
                                bool sign = (bits >> 31) & 1;
                                uint32_t e8 = (bits >> 23) & 0xFF;
                                uint32_t m23 = bits & 0x7FFFFF;

                                uint8_t e4m3_val;
                                
                                if (e8 == 0 && m23 == 0) {
                                    /* ±Zero */
                                    e4m3_val = sign ? 0x80 : 0x00;
                                } else if (e8 == 255) {
                                    /* NaN / Inf → 饱和 */
                                    e4m3_val = sign ? 0xFE : 0x7E;  // ±448 saturate
                                }  else {
                                    int32_t true_exp;
                                    uint32_t eff_mant; 
                                    if (e8 == 0) {
                                        /* FP32 subnormal → normalize */
                                        int shift = __builtin_clz(m23) - 8;
                                        eff_mant = m23 << (shift + 1);
                                        true_exp = -126 - shift;
                                    } else {
                                        /* Normal */
                                        eff_mant = (1U << 23) | m23;
                                        true_exp = (int32_t)e8 - 127;
                                    }
                                    int32_t e4m3_exp = true_exp + 7; 
                                    if (e4m3_exp <= 0) {
                                        if (e4m3_exp <= -4) {
                                            e4m3_val = sign ? 0x80 : 0x00;
                                        } else {
                                           int shift_right = 1 - e4m3_exp;  // 正值: 需要右移的位数
                                            /* 右移后的值: (eff_mant >> shift_right) 中,
                                            * bit[22:20] 对应非规格化的 3 位尾数 */
                                            if (shift_right >= 24) {
                                                e4m3_val = sign ? 0x80 : 0x00;
                                            } else {
                                                uint32_t sub_mant = (eff_mant >> (shift_right - 1)) & 0x7;
                                                /* 简单截断 (RTZ); 如需舍入可在此处加 guard/round/sticky */
                                                e4m3_val = (sign << 7) | sub_mant;
                                                /* 如果 sub_mant == 0 且舍入后为零，保持符号位区分 ±0 */
                                            } 
                                        }
                                        
                                    } else if (e4m3_exp >= 15) {
                                        /* 饱和 */
                                        e4m3_val = (sign << 7) | 0x7E;  // exp=F, mant=111
                                    } else {
                                        /* 在范围内：截断尾数 24位 → 3位 */
                                        uint32_t m3 = (eff_mant >> 20) & 0x7;
                                        
                                        uint32_t guard  = (eff_mant >> 19) & 1;
                                        uint32_t round  = (eff_mant >> 18) & 1;
                                        uint32_t sticky = ((eff_mant & 0x3FFFF) != 0) ? 1 : 0;

                                        if (guard && (round || sticky)) {
                                            /* 大于中间值 → 向上舍入 */
                                            m3++;
                                        } else if (guard && !round && !sticky) {
                                            /* 恰好中间值 → 向偶数舍入 (RNE) */
                                            if (m3 & 1) {  // 当前尾数是奇数
                                                m3++;
                                            }
                                        }
                                        /* RTZ (guard=0): 不做任何操作 */

                                        /* 处理尾数进位: 7+1 = 8 → 尾数溢出 */
                                        if (m3 >= 8) {
                                            m3 = 0;
                                            e4m3_exp++;
                                            if (e4m3_exp >= 15) {
                                                /* 进位导致超出范围 → 饱和 */
                                                e4m3_val = (sign << 7) | 0x7E;
                                            } else {
                                                e4m3_val = (sign << 7) | ((uint32_t)e4m3_exp << 3) | m3;
                                            }
                                        } else {
                                            e4m3_val = (sign << 7) | ((uint32_t)e4m3_exp << 3) | m3;
                                        }
                                    }
                                }
                                if (rd != 0) lane->fpr[rd] = e4m3_val;
                                
                            }
                            break;
                            case 2: { /* fcvt.s.e5m2 fd, fs1 — E5M2 → FP32 (上行读) */
                                uint8_t e5m2_bits = lane->fpr[rs1] & 0xFF;

                                /* 提取 E5M2 位域: S(1) + E(5) + M(2) */
                                bool     sign  = (e5m2_bits >> 7) & 1;
                                uint8_t  e5    = (e5m2_bits >> 2) & 0x1F;  // 5-bit exponent
                                uint8_t  m2    = e5m2_bits & 0x3;           // 2-bit mantissa

                                uint32_t f32_val;

                                if (e5 == 0) {
                                    if (m2 == 0) {
                                        /* ±Zero */
                                        f32_val = sign ? 0x80000000U : 0x00000000U;
                                    } else {
                                        /* Subnormal: value = 0.MM * 2^(-14)
                                        * 只有 3 种可能值: M2=01,10,11
                                        *
                                        * 归一化: 找到 msb, 左移, 计算真实指数
                                        */
                                        int shift = __builtin_ctz(m2);  // m2=01→shift=0, m2=10→shift=1, m2=11→shift=0
                                        uint32_t norm_mantissa = m2 << shift;  // 把最高位移到 bit 1
                                        // norm_mantissa 现在形式为 10 或 11 (即 1.x)
                                        
                                        int32_t true_exp = -14 - shift;   // -16 是基础指数
                                        
                                        // 构造 FP32: 尾数部分去掉隐含的1(bit1)，剩余位左移到位
                                        uint32_t fp32_m = ((norm_mantissa >> 1) & 1) << 21;  // 剩余1位尾数放到 FP32 高位
                                        
                                        f32_val = ((uint32_t)sign << 31)
                                                | ((uint32_t)(true_exp + 127) << 23)
                                                | fp32_m;  // 隐含的1
                                    }
                                } else if (e5 == 31) {  // 0b11111
                                    if (m2 == 0) {
                                        /* ±Infinity */
                                        f32_val = sign ? 0xFF800000U : 0x7F800000U;
                                    } else {
                                        /* NaN (Quiet NaN) */
                                        f32_val = 0x7FC00000U;  // 标准 Quiet NaN
                                        if (sign) f32_val |= 0x80000000U;  // 可选: 保持符号
                                    }
                                } else {
                                    /* Normal: value = (1 + M2/4) * 2^(e5 - 15) */
                                    int32_t true_exp = (int32_t)e5 - 15;
                                    uint32_t f32_exp   = (uint32_t)(true_exp + 127);
                                    uint32_t f32_mant  = (uint32_t)m2 << 21;  // 2位尾数 → FP32 高2位

                                    f32_val = ((uint32_t)sign << 31)
                                            | (f32_exp << 23)
                                            | f32_mant| 0x40000000U;
                                }

                                if (rd != 0) lane->fpr[rd] = f32_val;
                            }
                            break;

                            case 3: { /* fcvt.e5m2.s fd, fs1 — FP32 → E5M2 (下行写) */
                            uint32_t fs1_val = lane->fpr[rs1];
                            
                            bool     sign  = (fs1_val >> 31) & 1;
                            uint8_t  exp32 = (fs1_val >> 23) & 0xFF;
                            uint32_t m23   = fs1_val & 0x7FFFFF;
                            
                            uint8_t e5m2_val;

                            if (exp32 == 0 && m23 == 0) {
                                /* ±Zero */
                                e5m2_val = sign ? 0x80 : 0x00;
                                
                            } else if (exp32 == 255) {
                                /* Inf / NaN → 保持为 E5M2 的 Inf/NaN */
                                if (m23 == 0) {
                                    /* Inf → E5M2 Inf (E=31, M=0) */
                                    e5m2_val = (sign << 7) | 0x7C;  // 0_11111_00 = Inf
                                } else {
                                    /* NaN → E5M2 NaN (E=31, M≠0) */
                                    e5m2_val = (sign << 7) | 0x7E;  // 0_11111_10 = NaN (Quiet)
                                }
                                
                            } else {
                                /* Normal or Subnormal FP32 */
                                int32_t true_exp;
                                uint32_t eff_mant;
                                
                                if (exp32 == 0) {
                                    /* FP32 Subnormal → normalize */
                                    int shift = __builtin_clz(m23) - 8;  // 32-23=9, clz范围9~31, 减8得shift
                                    eff_mant = m23 << (shift + 1);
                                    true_exp = -126 - shift;
                                } else {
                                    /* FP32 Normal */
                                    eff_mant = (1U << 23) | m23;  // 加上隐含的1 → 24位有效数
                                    true_exp = (int32_t)exp32 - 127;
                                }
                                
                                /* 转换到 E5M2 指数域: bias 127 → 15 */
                                int32_t e5m2_exp = true_exp + 15;
                                
                                if (e5m2_exp <= 0) {
                                    /* 下溢到 E5M2 零或非正规 */
                                    if (e5m2_exp <= -2) {  // 有效数全部移出
                                        e5m2_val = sign ? 0x80 : 0x00;  // → ±0
                                    } else {
                                        /* E5M2 非正规: 右移 eff_mant 提取 2 位尾数 */
                                        int shift_right = 1 - e5m2_exp;
                                        if (shift_right >= 24) {
                                            e5m2_val = sign ? 0x80 : 0x00;
                                        } else {
                                            uint32_t sub_m = (eff_mant >> (shift_right + 22)) & 0x3;
                                            // 舍入处理...
                                            e5m2_val = (sign << 7) | sub_m;  // E=0, M=sub_m
                                        }
                                    }
                                    
                                } else if (e5m2_exp >= 31) {
                                    /* 上溢 → 饱和到 E5M2 Inf */
                                    e5m2_val = (sign << 7) | 0x7C;  // 0_11111_00 = ±Inf
                                    
                                } else {
                                    /* 在范围内: 24位尾数 → 2位尾数 + 舍入 */
                                    uint32_t m2 = (eff_mant >> 21) & 0x3;  // 取高2位
                                    
                                    /* 舍入: guard(位20), round(位19), sticky(位18~0) */
                                    uint32_t guard  = (eff_mant >> 20) & 1;
                                    uint32_t round  = (eff_mant >> 19) & 1;
                                    uint32_t sticky = ((eff_mant & 0x7FFFF) != 0) ? 1 : 0;
                                    
                                    /* RNE 舍入逻辑 (与你的 E4M3 代码一致) */
                                    if (guard && (round || sticky)) {
                                        m2++;  // 大于中间值 → 向上舍入
                                    } else if (guard && !round && !sticky) {
                                        if (m2 & 1) { m2++; }  // 中间值 → 向偶数舍入
                                    }
                                    /* RTZ (guard=0): 不操作 */

                                    /* 处理进位: m2 从 3 溢出到 4 (即 2-bit 存不下) */
                                    if (m2 >= 4) {
                                        m2 = 0;
                                        e5m2_exp++;
                                        if (e5m2_exp >= 31) {
                                            e5m2_val = (sign << 7) | 0x7C;  // 进位溢出 → Inf
                                        } else {
                                            e5m2_val = (sign << 7) | ((uint8_t)e5m2_exp << 2) | m2;
                                        }
                                    } else {
                                        e5m2_val = (sign << 7) | ((uint8_t)e5m2_exp << 2) | m2;
                                    }
                                }
                            }
                            
                            if (rd != 0) lane->fpr[rd] = e5m2_val;
                            }
                            break;
                        }
                        update_fcsr(lane);
                        break;
                    case 0x26:
                        switch (rs2) {
                            case 0:{ /* fcvt.s.e2m1 fd, fs1 — E2M1 → FP32 */
                                uint8_t e2m1_bits = lane->fpr[rs1] & 0xF;
                                static const uint32_t e2m1_to_f32[16] = {
                                    /* 正数 (S=0) */
                                    [0x0] = 0x00000000U,              //  0.0f
                                    [0x1] = 0x3F000000U,              //  0.5f  (0 01111110 000...)
                                    [0x2] = 0x3F800000U,              //  1.0f  (0 01111111 000...)
                                    [0x3] = 0x3FC00000U,              //  1.5f  (0 01111111 100...)
                                    [0x4] = 0x40000000U,              //  2.0f  (0 10000000 000...)
                                    [0x5] = 0x40400000U,              //  3.0f  (0 10000000 100...)
                                    [0x6] = 0x40800000U,              //  4.0f  (0 10000001 000...)
                                    [0x7] = 0x40C00000U,              //  6.0f  (0 10000001 100...)
                                    
                                    /* 负数 (S=1): 符号位置 1 */
                                    [0x8] = 0x80000000U,              // -0.0f
                                    [0x9] = 0xBF000000U,              // -0.5f
                                    [0xA] = 0xBF800000U,              // -1.0f
                                    [0xB] = 0xBFC00000U,              // -1.5f
                                    [0xC] = 0xC0000000U,              // -2.0f
                                    [0xD] = 0xC0400000U,              // -3.0f
                                    [0xE] = 0xC0800000U,              // -4.0f
                                    [0xF] = 0xC0C00000U,              // -6.0f
                                };
                                if (rd != 0) lane->fpr[rd] = e2m1_to_f32[e2m1_bits];
                            }
                            break;
                            case 1:{ /* fcvt.e2m1.s fd, fs1 — FP32 → E2M1 */
                                uint32_t f32_val = lane->fpr[rs1];  // 例如 100.0f = 0x42C80000
                                bool   sign  = (f32_val >> 31) & 1;         // 符号
                                uint8_t exp32 = (f32_val >> 23) & 0xFF;     // 指数
                                uint32_t m23   = f32_val & 0x7FFFFF;
                                uint32_t abs_val = f32_val & 0x7FFFFFFFU;
                                uint8_t e2m1_result;
                                if (exp32 == 0 && m23 == 0) {
                                    e2m1_result = sign ? 0x8 : 0x0;
                                } else if (exp32 == 255) {
                                    e2m1_result = sign ? 0xF : 0x7;
                                } else {
                                    /*
                                    * ★ 核心手写逻辑: 用 FP32 位模式比较来做量化
                                    * 
                                    * 方法: 将 abs_val 与各 E2M1 代表值的 FP32 阈值进行比较,
                                    *       找到落在哪个区间, 然后按舍入模式选择
                                    *
                                    * 区间划分 (正半轴):
                                    *   [0,     0.25)     → 0    (但 0 单独处理了)
                                    *   [0.25,  0.75)     → 0.5  (中点在 0.5, 即 0.25~0.75 选 0.5)
                                    *   [0.75,  1.25)     → 1.0
                                    *   [1.25,  1.75)     → 1.5
                                    *   [1.75,  2.5)      → 2.0
                                    *   [2.5,   3.5)      → 3.0
                                    *   [3.5,   5.0)      → 4.0
                                    *   [5.0,   +∞)       → 6.0  (饱和!)
                                    */
                                    
                                    // ★ 方法A: 逐级比较 (最直观, 易调试)
                                    if (abs_val < 0x3D800000U) {     
                                        
                                        e2m1_result = 0x0;  // 0 or 0.5
                                    
                                    }
                                    else if (abs_val < 0x3F400000U) {  // < 0.5
                                        e2m1_result = 0x1;  // 0.5
                                    }
                                    else if (abs_val < 0x3FA00000U) {  // ~1.2 (1.0 和 1.5 之间的判定点)
                                        // 在 0.5 ~ 1.5 中点附近: 需要看具体舍入
                                        e2m1_result = 0x2;  // 1.0 (默认 RTZ/RNE)
                                    }
                                    else if (abs_val < 0x3FE00000U) {  // ~2.25
                                        e2m1_result = 0x3;  // 1.5
                                    }
                                    else if (abs_val < 0x40200000U) {  // ~3.75
                                        e2m1_result = 0x4;  // 2.0
                                    }
                                    else if (abs_val < 0x40600000U) {  // ~5.5
                                        e2m1_result = 0x5;  // 3.0
                                    }
                                    else if (abs_val < 0x40A00000U) {  // ~7 (超过6的一半?)
                                        e2m1_result = 0x6;  // 4.0
                                    }
                                    else {
                                        // ★ 超过最大值范围 → 饱和到 6.0!
                                        e2m1_result = 0x7;  // 6.0 (E2M1 max)
                                    }
                                }
                                /* 加回符号位 */
                                if (sign) e2m1_result |= 0x8;
                                if (rd != 0) lane->fpr[rd] = e2m1_result;
                            }
                            break;
                        }
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
                                    if (rd !=0) lane->gpr[rd] =
                                    
                                    0;
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
            return s->kernel.block_dim[0]; /* block_dim.X */
        case 0x24:
            return s->kernel.block_dim[1]; /* block_dim */   
        case 0x28:
            return s->kernel.block_dim[2]; /* block_dim.x */

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

