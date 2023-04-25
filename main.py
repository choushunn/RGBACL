import time

from fun import Fun_GeneratingPhase, Fun_Diffra2DAngularSpectrum_BerryPhase, Fun_EfieldParameters, \
    Fun_UpdateParticleSingletBerryPhase, Fun_GeneRandomGeneration, CPSWFs_FWHM_calculation, Fun_FitnessLenserror, \
    Fun_GeneRandomGenerationSinglet

try:
    import cupy as np
except ModuleNotFoundError as e:
    print(e)
    import numpy as np

from tqdm import tqdm
import pandas as pd

print(np.__version__)


class ACL:
    def __init__(self):
        self.check_env()
        # 波长
        self.lam = np.array([0.5328, 0.6328, 0.7328])
        self.lamc = self.lam[1]
        self.n_lam = self.lam.size
        # 光速
        self.c = 0.3
        # 焦距
        self.FocalLength = 61 * self.lamc
        # 外径
        self.R_outter = 100 * self.lamc
        # 内径
        self.R_inner = 0 * self.lamc
        # Z轴范围
        self.Zrange = 30 * self.lamc
        self.NA = np.sin(np.arctan(self.R_outter / self.FocalLength))
        self.diff_limit = 0.5 * self.lamc / self.NA
        self.P_metasurface = 0.4
        self.Nr_inner = np.ceil(self.R_inner / self.P_metasurface)
        self.Nr_outter = int(np.floor(self.R_outter / self.P_metasurface) + 1)
        # 每行/列正方形的数量 floor:取小于该数的整数
        self.N = np.floor(2 * self.R_outter / self.P_metasurface)
        self.a = np.mod(self.N, 2)
        self.r = np.arange(0, (self.Nr_outter - 1) * self.P_metasurface + self.P_metasurface, self.P_metasurface)
        self.mm = self.r.size
        self.R0 = (self.Nr_outter - 1) * self.P_metasurface
        self.phi_lamc = 2 * np.pi * (
                np.sqrt(self.R0 ** 2 + self.FocalLength ** 2) - np.sqrt(
            self.r ** 2 + self.FocalLength ** 2)) / self.lamc
        self.GD = (np.sqrt(self.R0 ** 2 + self.FocalLength ** 2) - np.sqrt(
            self.r ** 2 + self.FocalLength ** 2)) / self.c  # 将GD全部转化为正值 fs
        self.Nr_gene = 1
        self.Nz = 50  # 计算焦平面前后2*Nz+1个平面内的光场(Nz==0时，只计算焦平面上的光场)
        self.GDmax = 5
        self.GDR = np.zeros((self.Nr_outter))
        j = 0
        for i in range(self.Nr_outter):
            if self.GDmax - self.GD[j] + self.GD[i] > 0:
                self.GDR[i] = self.GDmax - self.GD[j] + self.GD[i]
            else:
                j = i
                self.GDR[i] = self.GDmax
                self.Nr_gene = self.Nr_gene + 1
        self.GDR = self.GDR.T
        self.N_particle = 6  # 粒子的数量
        self.c1 = 2
        self.c2 = 2  # 用于计算粒子速度的系数 2，2
        self.w = 0.5  # 计算粒子速度的权重0.5

        self.Vmax = 1
        self.Vmin = -1
        self.N_gene = 2 * np.pi  # 相位不连续性△φ（λc）的边界

        self.Max_iteration = 100  # 迭代次数
        self.Fitness_GlobalBest = 500000

    def run(self):
        # 放当前的参数：半高全宽、旁瓣比、强度(多波长)
        FWHM_PersonalPresent0 = np.zeros((self.n_lam))
        SideLobeRatio_PersonalPresent0 = np.zeros((self.n_lam))
        IntensPeak_PersonalPresent0 = np.zeros((self.n_lam))
        Focal_offset_PersonalPresent0 = np.zeros((self.n_lam))
        # 存放当前的参数：半高全宽、旁瓣比、强度
        FWHM_PersonalPresent = np.zeros((self.N_particle))
        SideLobeRatio_PersonalPresent = np.zeros((self.N_particle))
        IntensPeak_PersonalPresent = np.zeros((self.N_particle))
        Focal_offset_PersonalPresent = np.zeros((self.N_particle))
        # 随机生成初始化值
        Gene_PersonalPresent, Fitness_PersonalPresent, Fitness_PersonalBest, Gene_PersonalBest, Velocity_PersonalPresent = Fun_GeneRandomGeneration(
            self.N_gene, self.Nr_gene, self.N_particle, self.Max_iteration)
        Gene_GlobalBest = Gene_PersonalPresent[self.N_particle - 10, :]
        # 衍射计算参数
        SpotType = 0  # 0-solid focal spot; 1-hollow focal spot
        TargetFWHM = 0.5
        TargetSidlobe = 0.25
        TargetPeakIntensity = 100000
        TargetFocal_offset = 0.0001
        FieldOfView = 10 * self.lamc
        TargetFieldPolar = 2  # 0-Transverse Polar; 1-Longitudinal Polar; 2-All Polarization components
        N_sampling = 1024
        n_refra0 = 1.46  # refractive index of the medium before lens
        n_refra1 = 1  # refractive index of the medium after lens
        N_Phase = 32  # 相位的数目!!!!!!必须与所提供的超表面结构个数一致  因为在Fun_Diffra2DAngularSpectrum中相位个数是按照超表面结构的个数提供的
        R_calculation = self.R_outter
        Dx = 2 * R_calculation / N_sampling
        # 局部最优
        Gene_LensPersonalBestL = np.zeros((self.Nr_gene))  # 局部社会因子
        FWHM_PersonalBest = np.zeros((self.Max_iteration))
        SideLobeRatio_PersonalBest = np.zeros((self.Max_iteration))
        IntensPeak_PersonalBest = np.zeros((self.Max_iteration))
        Focal_offset_PersonalBest = np.zeros((self.Max_iteration))
        # 全局最好
        FWHM_GlobalBest = np.zeros(self.Max_iteration)
        SideLobeRatio_GlobalBest = np.zeros(self.Max_iteration)
        IntensPeak_GlobalBest = np.zeros(self.Max_iteration)
        Focal_offset_GlobalBest = np.zeros(self.Max_iteration)
        if self.Nz > 0:
            dZ = self.Zrange / (2 * self.Nz)
        else:
            dZ = 0
        Zd = np.array([self.FocalLength + dZ * (-self.Nz)])
        for k in range(-self.Nz + 1, self.Nz + 1):
            Zd = np.hstack([Zd, self.FocalLength + dZ * k])

        nn = 300  # 显示区域范围,单个格子同网格大小一致
        XX = ((np.arange(N_sampling / 2 - nn, N_sampling / 2 + 1 + nn) - N_sampling / 2 - 0.5) * Dx)
        XX_Itotal_Ir_Iphi_IzDisplay = np.zeros((2 * nn + 2, 2 * self.Nz + 1))  # 存放不同传播面上总场数据
        # 开始迭代
        for n_iteration in range(self.Max_iteration):
            pbar = tqdm(range(self.N_particle))
            for k in pbar:
                pbar.set_description(f"{n_iteration + 1}/{self.Max_iteration}")
                # 返回三个相位
                phase = Fun_GeneratingPhase(self.GDR, Gene_PersonalPresent[k], self.Nr_gene, self.Nr_outter, self.lam,
                                            self.r, self.FocalLength, self.c)
                DOF = np.zeros(self.n_lam)
                Intensity_sum = np.zeros(self.n_lam)
                # 循环三个通道
                for i in range(self.n_lam):
                    wavelength = self.lam[i]
                    # 当前计算的波长对应相位
                    phasei = phase[i, :]
                    Intensity = np.zeros((self.Nr_outter, 2 * self.Nz + 1))
                    IPeak = np.zeros((2 * self.Nz + 1))
                    # 计算不同传播面
                    for nnz in range(2 * self.Nz + 1):
                        Ex, Ey, Ez = Fun_Diffra2DAngularSpectrum_BerryPhase(wavelength, Zd[nnz], self.R_outter,
                                                                            self.R_inner, Dx,
                                                                            N_sampling, n_refra1, self.P_metasurface,
                                                                            phasei,
                                                                            self.Nr_outter, nn)
                        if TargetFieldPolar == 0:  # Transvers components
                            Intensity = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
                        elif TargetFieldPolar == 1:  # Longitudinal components
                            Intensity = np.abs(Ez) ** 2
                        elif TargetFieldPolar == 2:  # All components
                            Intensity = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
                        XX_Itotal_Ir_Iphi_IzDisplay[:, nnz] = Intensity[:, nn + 1]
                        IPeak[nnz] = np.max(np.max(XX_Itotal_Ir_Iphi_IzDisplay[:, nnz]))
                    # 传播面上光场计算结束
                    DOF[i] = CPSWFs_FWHM_calculation(IPeak, Zd / self.lamc, self.Nz)  # 计算传播面上的焦深
                    Intensity_sum[i] = np.sum(
                        IPeak[(2 * self.Nz + 1) // 3:2 * (2 * self.Nz + 1) // 3])  # 取中间 DOF 区域的强度和
                    IPeakmax = np.max(IPeak)
                    In = np.where(IPeak == IPeakmax)[-1]  # 找到最大强度对应的位置平面,取最后一个 In 中的元素
                    Intensity_z = XX_Itotal_Ir_Iphi_IzDisplay[:, In]
                    Nxc, Nyc = np.where(np.abs(Intensity_z - IPeakmax) < 10)
                    # if Nxc[0] == 0:
                    #     Nxc[0] = nn + 1
                    FWHM_x, SideLobeRatio_x, IntensPeak_x = Fun_EfieldParameters(Intensity_z, XX.T, Nxc[0], SpotType)
                    FWHM_PersonalPresent0[i] = FWHM_x / self.lam[i]
                    IntensPeak_PersonalPresent0[i] = IntensPeak_x
                    SideLobeRatio_PersonalPresent0[i] = SideLobeRatio_x
                    Focal_offset_PersonalPresent0[i] = np.abs(Zd[In] - self.FocalLength) / self.FocalLength  # 焦距的偏移量
                    # 计算波长结束
                FWHM_PersonalPresent[k] = np.max(FWHM_PersonalPresent0)
                IntensPeak_PersonalPresent[k] = np.min(IntensPeak_PersonalPresent0)
                SideLobeRatio_PersonalPresent[k] = np.max(SideLobeRatio_PersonalPresent0)
                Focal_offset_PersonalPresent[k] = np.max(Focal_offset_PersonalPresent0)
                Fitness_PersonalPresent[k] = Fun_FitnessLenserror(TargetFWHM, TargetSidlobe, TargetPeakIntensity,
                                                                  TargetFocal_offset, FWHM_PersonalPresent[k],
                                                                  SideLobeRatio_PersonalPresent[k],
                                                                  IntensPeak_PersonalPresent[k],
                                                                  Focal_offset_PersonalPresent[k], DOF, Intensity_sum)

                # 将以下数据写入 excel
                # Focaloffset_Fitness_FWHM_IntensPeak= [Focal_offset_PersonalPresent0, Fitness_PersonalPresent[k], FWHM_PersonalPresent[k], IntensPeak_PersonalPresent[k],SideLobeRatio_PersonalPresent[k],DOF,Intensity_sum]
                # Gene_Personal = Gene_PersonalPresent[k, :]
            # update personal best
            C, I = np.min(Fitness_PersonalPresent), np.argmin(Fitness_PersonalPresent)
            Gene_PersonalBest = Gene_PersonalPresent[I, :]
            mL, nL = np.min(Fitness_PersonalPresent[self.N_particle - 15:self.N_particle - 10]), np.argmin(
                Fitness_PersonalPresent[self.N_particle - 15:self.N_particle - 10])
            Gene_LensPersonalBestL = Gene_PersonalPresent[nL + self.N_particle - 15, :]
            Fitness_PersonalBest[n_iteration] = Fitness_PersonalPresent[I]
            FWHM_PersonalBest[n_iteration] = FWHM_PersonalPresent[I]
            SideLobeRatio_PersonalBest[n_iteration] = SideLobeRatio_PersonalPresent[I]
            IntensPeak_PersonalBest[n_iteration] = IntensPeak_PersonalPresent[I]
            Focal_offset_PersonalBest[n_iteration] = Focal_offset_PersonalPresent[I]
            NO = 1  # 记录全局最优
            if C < self.Fitness_GlobalBest:
                FWHM_GlobalBest[NO] = FWHM_PersonalPresent[I]
                SideLobeRatio_GlobalBest[NO] = SideLobeRatio_PersonalPresent[I]
                IntensPeak_GlobalBest[NO] = IntensPeak_PersonalPresent[I]
                Focal_offset_GlobalBest[NO] = Focal_offset_PersonalPresent[I]
                self.Fitness_GlobalBest = C
                Gene_GlobalBest = Gene_PersonalPresent[I, :]
                NO = NO + 1
                Gene_PersonalPresent, Velocity_PersonalPresent = Fun_UpdateParticleSingletBerryPhase(
                    Gene_PersonalPresent,
                    Gene_PersonalBest,
                    Gene_GlobalBest,
                    Gene_LensPersonalBestL,
                    Velocity_PersonalPresent,
                    self.N_gene, self.N_particle,
                    self.Nr_gene)
            # print(Velocity_PersonalPresent.shape)
            # if n_iteration % 50 == 0:
            #     B, I = np.sort(Fitness_PersonalBest)[::-1], np.argsort(Fitness_PersonalBest)[::-1]
            #     for i in range(self.N_particle // 5):
            #         k = I[i]
            #         # print(Gene_PersonalPresent.shape,Fitness_PersonalPresent.shape,Fitness_PersonalBest.shape,Gene_PersonalBest.shape,Velocity_PersonalPresent.shape)
            #         Gene_PersonalPresent[k, :], Fitness_PersonalPresent[k], Fitness_PersonalBest[k], Gene_PersonalBest[
            #                                                                                          k], Velocity_PersonalPresent[
            #                                                                                              k,
            #                                                                                              :] = Fun_GeneRandomGenerationSinglet(
            #             self.N_gene, self.Nr_gene)

            phase = Fun_GeneratingPhase(self.GDR, Gene_GlobalBest, self.Nr_gene, self.Nr_outter, self.lam, self.r,
                                        self.FocalLength, self.c)

            # 输出当前迭代结果
            df = pd.DataFrame({"n_iteration": [n_iteration],
                               "FWHM_GlobalBest": [FWHM_GlobalBest.get()],
                               "SideLobeRatio_GlobalBest": [SideLobeRatio_GlobalBest.get()],
                               "IntensPeak_GlobalBest": [IntensPeak_GlobalBest.get()],
                               "Focal_offset_GlobalBest": [Focal_offset_GlobalBest.get()],
                               "Fitness_GlobalBest": [self.Fitness_GlobalBest]
                               })
            self.save_results(df)

            # mdic = {'lam': self.lam,
            #         'FocalLength': self.FocalLength,
            #         'R_outter': self.R_outter,
            #         'R_inner': self.R_inner,
            #         'P_metasurface': self.R_inner,
            #         'SpotType': self.R_inner,
            #         'TargetFWHM': self.R_inner,
            #         'FWHM_GlobalBest': self.R_inner,
            #         'SideLobeRatio_GlobalBest': self.R_inner,
            #         'TargetSidlobe': self.R_inner,
            #         'TargetPeakIntensity': self.R_inner,
            #         'IntensPeak_GlobalBest': self.R_inner,
            #         'FieldOfView': self.R_inner,
            #         'TargetFieldPolar': self.R_inner,
            #         'Focal_offset_GlobalBest': self.R_inner,
            #         'N_sampling': self.R_inner,
            #         'n_refra0': self.R_inner,
            #         'n_refra1': self.R_inner,
            #         'N_Phase': self.R_inner,
            #         'R_calculation': self.R_inner,
            #         'Dx': self.R_inner,
            #         'Nr_inner': self.R_inner,
            #         'Nr_outter': self.R_inner,
            #         'Gene_GlobalBest': self.R_inner,
            #         'phase_best': self.R_inner,
            #         'Fitness_GlobalBest': self.R_inner
            #         }

            # 保存为文件

    def save_results(self, df):
        df.to_csv(self.filename_SaveData + ".csv", mode='a', header=False, index=False)
        df.to_excel(self.filename_SaveData + '.xlsx', sheet_name='Sheet1', header=False, index=False)

    def check_env(self):
        import os
        import datetime
        # 设置保存结构文件夹名称
        resultFolderName = 'results'
        # 设置保存的文件名
        now = datetime.datetime.now()
        self.filename_SaveData = f"results/save_{now.strftime('%Y-%m-%d_%H-%M-%S')}"

        # 检查文件夹是否存在
        if not os.path.exists(resultFolderName):
            # 如果文件夹不存在，则创建该文件夹
            os.mkdir(resultFolderName)


if __name__ == '__main__':
    acl = ACL()
    acl.run()
