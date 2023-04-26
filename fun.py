import cmath

import matplotlib.pyplot as plt

try:
    import cupy as np
except ModuleNotFoundError as e:
    import numpy as np


def Fun_GeneratingPhase(GDR, Gene_phase, Nr_gene, Nr_outter, lam, r, FocalLength, c):
    """
    生成相位
    :param GDR:
    :param Gene_PersonalPresent:
    :param Nr_gene:
    :param Nr_outter:
    :param lam:波长
    :param r:
    :param FocalLength:焦距
    :param c:光速
    :return:phase,三个波长的相位
    """
    lamc = lam[1]
    # phase = np.zeros((3, 159))
    GDRmax = np.max(GDR)
    p = np.where(GDR == GDRmax)[-1]
    # print(p, type(p))
    phasec = np.zeros((Nr_outter))
    rr = np.zeros(((Nr_gene)))
    L = np.zeros(((Nr_gene)))
    # Gene_phase = np.zeros((Nr_gene))
    for j in range(Nr_gene):
        # print(j)
        if j < Nr_gene - 1:
            # print(p[j])
            rr[j] = r[p[j]]
            L[j] = np.sqrt(rr[j] ** 2 + FocalLength ** 2)
            # print(p[j],j)
            phasec[p[j]:p[j + 1]] = 2 * np.pi / lamc * (np.sqrt(r[p[j + 1]] ** 2 + FocalLength ** 2) - np.sqrt(
                r[p[j]:p[j + 1]] ** 2 + FocalLength ** 2))
            if j == 0:
                rr[j] = 0
                L[j] = np.sqrt(rr[j] ** 2 + FocalLength ** 2)
            phasec[p[j]:p[j + 1]] = phasec[p[j]:p[j + 1]] - 2 * np.pi / lamc * (L[j] - L[0]) + Gene_phase[j]
        else:
            rr[j] = r[p[j]]
            L[j] = np.sqrt(rr[j] ** 2 + FocalLength ** 2)
            phasec[p[j]:Nr_outter] = 2 * np.pi / lamc * (np.sqrt(r[Nr_outter - 1] ** 2 + FocalLength ** 2) - np.sqrt(
                r[p[j]:Nr_outter] ** 2 + FocalLength ** 2))
            phasec[p[j]:Nr_outter] = phasec[p[j]:Nr_outter] - 2 * np.pi / lamc * (L[j] - L[0]) + Gene_phase[j]
    # print( phasec)
    # plt.plot(phasec.get())
    # plt.show()
    phasec = np.transpose(phasec)
    phase0 = phasec + GDR * (2 * np.pi * c * (1 / lam[0] - 1 / lamc))
    phase1 = phasec + GDR * (2 * np.pi * c * (1 / lam[1] - 1 / lamc))
    phase2 = phasec + GDR * (2 * np.pi * c * (1 / lam[2] - 1 / lamc))
    # phase[0, :] = phase0
    # phase[1, :] = phase1
    # phase[2, :] = phase2
    return [phase0, phase1, phase2]


def FourrierTrans2D(g, Dx, N, flag):
    """

    :param g:
    :param Dx:
    :param N:
    :param flag:
    :return:
    """
    num = np.arange(N)
    a = np.exp(1j * 2 * np.pi / N * (N / 2 - 0.5) * num)
    A = a.reshape(-1, 1) * a
    C = np.exp(-1j * 2 * np.pi / N * (N / 2 - 0.5) ** 2 * 2) * A

    if flag == 1:
        return Dx ** 2 * C * np.fft.fft2(A * g)
        # G = np.fft.fft2(A * g)  # 二维FFT变换后会发生错位
        # G = np.roll(G, -1, axis=0)  # 先进行错位纠正
        # G = Dx**2 * C * G  # 再进行二维FFT计算结果的系数修正
    if flag == -1:
        return (1. / (N * Dx)) ** 2 * N ** 2 * np.conj(C) * np.fft.ifft2(np.conj(A) * g)

        # 二维FFT正变换进行错位修正后，反变换时不需再进行错位修正
    # G = np.roll(G, -1, axis=0)  # 修补，没解决根本问题(为什么傅里叶变换后会错位)07-01-2018尚待找到原因


def Diffraction2DTransPolar(Ex0, Ey0, Z, Wavelen0, n_refr, Dx, N, nn):
    """

    :param Ex0:
    :param Ey0:
    :param Z:
    :param Wavelen0:
    :param n_refr:
    :param Dx:
    :param N:
    :param nn:
    :return:
    """
    # num = np.arange(N)
    # freq = 1. / (N * Dx) * (num - N / 2 + 0.5)
    # freq_x = freq.T
    # freq_y = freq_x
    # n_refr_wave = (n_refr / Wavelen0) ** 2 * np.ones((N))
    # # fz 计算有误
    # fz = np.sqrt(n_refr_wave - freq_x ** 2 - freq_y ** 2)
    # SpectrumX = FourrierTrans2D(Ex0, Dx, N, 1)
    # SpectrumY = FourrierTrans2D(Ey0, Dx, N, 1)
    # SpectrumZ = -(freq_x * SpectrumX + freq_y * SpectrumY) / fz * np.exp(1j * 2 * np.pi * fz * Z)
    # SpectrumX = SpectrumX * np.exp(1j * 2 * np.pi * fz * Z)
    # SpectrumY = SpectrumY * np.exp(1j * 2 * np.pi * fz * Z)
    # Ex = FourrierTrans2D(SpectrumX, Dx, N, -1)
    # Ey = FourrierTrans2D(SpectrumY, Dx, N, -1)
    # Ez = FourrierTrans2D(SpectrumZ, Dx, N, -1)
    # Ex = Ex[N // 2 - nn:N // 2 + 2 + nn, N // 2 - nn:N // 2 + 2 + nn]  # 显示区域范围, 单个格子同网格大小一致
    # Ey = Ey[N // 2 - nn:N // 2 + 2 + nn, N // 2 - nn:N // 2 + 2 + nn]
    # Ez = Ez[N // 2 - nn:N // 2 + 2 + nn, N // 2 - nn:N // 2 + 2 + nn]

    num = np.arange(N)
    freq = 1. / (N * Dx) * (num - N / 2 + 0.5)
    freq_x = np.outer(freq, np.ones(N))
    freq_y = freq_x.T
    fza = ((n_refr / Wavelen0) ** 2 - freq_x ** 2 - freq_y ** 2).astype(np.complex128)
    fz = np.sqrt(fza)
    SpectrumX = FourrierTrans2D(Ex0, Dx, N, 1)
    SpectrumY = FourrierTrans2D(Ey0, Dx, N, 1)
    SpectrumZ = -(freq_x * SpectrumX + freq_y * SpectrumY) / fz * np.exp(1j * 2 * np.pi * fz * Z)
    SpectrumX = SpectrumX * np.exp(1j * 2 * np.pi * fz * Z)
    SpectrumY = SpectrumY * np.exp(1j * 2 * np.pi * fz * Z)
    Ex = FourrierTrans2D(SpectrumX, Dx, N, -1)
    Ey = FourrierTrans2D(SpectrumY, Dx, N, -1)
    Ez = FourrierTrans2D(SpectrumZ, Dx, N, -1)
    nn = int(nn)
    Ex = Ex[int(N / 2 - nn):int(N / 2 + 2 + nn), int(N / 2 - nn):int(N / 2 + 2 + nn)]
    Ey = Ey[int(N / 2 - nn):int(N / 2 + 2 + nn), int(N / 2 - nn):int(N / 2 + 2 + nn)]
    Ez = Ez[int(N / 2 - nn):int(N / 2 + 2 + nn), int(N / 2 - nn):int(N / 2 + 2 + nn)]
    # Exyz = np.zeros((3, 2 * nn + 2, 2 * nn + 2), dtype=np.complex128)

    return Ex, Ey, Ez


def Fun_Diffra2DAngularSpectrum_BerryPhase(Wavelen0, Zd, R_outter, R_inner, Dx, N, n_refra, P_metasurface, Gene_Lens0,
                                           Nring, nn):
    """

    :param Wavelen0:
    :param Zd:
    :param R_outter:
    :param R_inner:
    :param Dx:
    :param N:
    :param n_refra:
    :param P_metasurface:
    :param Gene_Lens0:
    :param Nring:
    :param nn:
    :return:
    """
    # i_IMG = 1j
    num = np.arange(N)
    N_sampling = N
    DT = 0  # 偶数
    # 产生X，Y矩阵
    Y = Dx * np.ones((N, 1)) * (num - N / 2 + 0.5)
    X = Y.transpose()

    # 计算每个采样点处的等效距离，即所属超表面结构单元中心位置到透镜中心的距离
    Rij = np.sqrt(np.ceil((np.abs(X) - 0.5 * P_metasurface) / P_metasurface) ** 2 + np.ceil(
        (np.abs(Y) - 0.5 * P_metasurface) / P_metasurface) ** 2) * P_metasurface
    # 确定每个采样点（超表面单元）对应的基因，在Gene_Lens中对应的基因片段（即对应基因片段的位置（第几个“环带”））；
    GeneN_ij = np.floor(Rij / P_metasurface) + 1
    GeneN_ij[GeneN_ij > Nring] = Nring

    # 给出每个采样点所属超表面单元的相位
    Phase_ijUnit1 = Gene_Lens0[GeneN_ij.astype(int) - 1]  # Gene(1:Nring)为相位；
    AmpProfile_ij1 = np.ones((N, N)) * 0.66
    AmpProfile_ij1[Rij >= R_outter] = 0
    Phase_ijUnit1[Rij >= R_outter] = 0
    Phase_ijUnit1[Rij < R_inner] = 0
    ####扩充器件边缘的非结构区域#######################
    Phase_ijUnit = np.zeros((N_sampling + DT, N_sampling + DT))
    AmpProfile_ij = np.zeros((N_sampling + DT, N_sampling + DT))
    Phase_ijUnit[DT // 2:N_sampling + DT // 2, DT // 2:N_sampling + DT // 2] = Phase_ijUnit1[0:N_sampling, 0:N_sampling]
    AmpProfile_ij[DT // 2:N_sampling + DT // 2, DT // 2:N_sampling + DT // 2] = AmpProfile_ij1[0:N_sampling,
                                                                                0:N_sampling]

    # --计算器件出射场-------------------------------
    Ex0 = AmpProfile_ij * np.exp(1j * Phase_ijUnit)
    Ey0 = AmpProfile_ij * np.exp(1j * (Phase_ijUnit + np.pi / 2))

    Ex, Ey, Ez = Diffraction2DTransPolar(Ex0, Ey0, Zd, Wavelen0, n_refra, Dx, N + DT, nn)

    return Ex, Ey, Ez


def CPSWFs_FWHM_calculation(IntensX, x, X_Center):
    """

    :param IntensX:
    :param x:
    :param X_Center:
    :return:
    """
    # 数据类型为行向量
    nX = IntensX.size
    Imax = IntensX[X_Center]
    Xfwhm1 = 0
    Xfwhm2 = 0
    flag = 0
    for i in range(X_Center):
        if IntensX[i] < 0.5 * Imax:
            if flag == 0:
                x1 = x[i + 1]
                I1 = IntensX[i + 1]
                x2 = x[i]
                I2 = IntensX[i]
                b = (I2 - I1) / (x2 - x1)
                c = I2 - b * x2
                Xfwhm1 = (0.5 * Imax - c) / b
                flag = 1

    flag = 0
    for i in range(X_Center, nX):
        if IntensX[i] < 0.5 * Imax:
            if flag == 0:
                x1 = x[i - 1]
                I1 = IntensX[i - 1]
                x2 = x[i]
                I2 = IntensX[i]
                b = (I2 - I1) / (x2 - x1)
                c = I2 - b * x2
                Xfwhm2 = (0.5 * Imax - c) / b
                flag = 1

    FWHM = Xfwhm2 - Xfwhm1
    return FWHM


def Fun_EfieldParameters(Intensity, X, Center, SpotType, Nr_outter):
    """

    :param Intensity:
    :param X:
    :param Center:
    :param SpotType:
    :return:
    """
    Intensity = Intensity.T[0]
    N = Intensity.size
    FWHM = 0
    SideLobeRatio = 0
    IntensPeak = 0
    if SpotType == 0:  # solid focal spot
        # calculating Peak Intensity
        IntensPeak = Intensity[Center]

        # calculating FWHM
        k1 = Center
        while Intensity[k1] > IntensPeak / 2 and k1 > 2:
            k1 = k1 - 1
        k2 = Center + 1
        while Intensity[k2] > IntensPeak / 2 and k2 < N - 1:
            k2 = k2 + 1
        x1 = X[k1]
        y1 = Intensity[k1]
        x2 = X[k1 + 1]
        y2 = Intensity[k1 + 1]
        b = (y2 - y1) / (x2 - x1)
        c = y2 - b * x2
        xL = (IntensPeak / 2 - c) / b

        x1 = X[k2]
        y1 = Intensity[k2]
        x2 = X[k2 - 1]
        y2 = Intensity[k2 - 1]
        b = (y2 - y1) / (x2 - x1)
        c = y2 - b * x2
        xR = (IntensPeak / 2 - c) / b

        FWHM = np.abs(xR - xL)

        # calculating Sidelobe
        k1 = Center
        while not ((Intensity[k1 - 1] >= Intensity[k1] and Intensity[k1] < Intensity[k1 + 1]) or k1 > 2):
            k1 = k1 - 1
        if k1 == Center:
            k1 = 1
        k2 = Center + 1
        while not (
                (Intensity[k2 - 1] > Intensity[k2] and Intensity[k2] <= Intensity[k2 + 1]) or k2 < N - 1):
            k2 = k2 + 1
        if k2 == Center + 1:
            k2 = N
        if k1 <= 0:
            k1 = 1
        if k2 > Intensity.size:
            k2 = Intensity.size - 1
        # max_index = np.argmax(Intensity)
        # if max_index < Nr_outter:
        #     max_index = Nr_outter
        # SideLobeRatio = np.max(Intensity[Nr_outter:]) / np.max(Intensity)
        # SideLobeRatio =
        # print(k1,k2,N)
        SideLobeRatio = max(np.max(Intensity[0:k1]), np.max(Intensity[k2 - 1:N])) / IntensPeak
    elif SpotType == 1:  # hollow focal spot
        # calculation peak
        k1 = Center
        while not (Intensity[k1 - 1] <= Intensity[k1] and Intensity[k1] > Intensity[k1 + 1] and k1 > 2):
            k1 = k1 - 1
        k2 = Center + 1
        while not (Intensity[k2 - 1] < Intensity[k2] and Intensity[k2] >= Intensity[k2 + 1] and k2 < N - 1):
            k2 = k2 + 1

        IntensPeak = max(Intensity[k1], Intensity[k2])

        # calculation FWHM
        kc1 = k1
        kc2 = k2
        while Intensity[k1] > IntensPeak / 2 and k1 < N - 1:
            k1 = k1 + 1
        while Intensity[k2] > IntensPeak / 2 and k2 > 2:
            k2 = k2 - 1

        x1 = X[k1]
        y1 = Intensity[k1]
        x2 = X[k1 - 1]
        y2 = Intensity[k1 - 1]
        b = (y2 - y1) / (x2 - x1)
        c = y2 - b * x2
        xL = (IntensPeak / 2 - c) / b

        x1 = X[k2]
        y1 = Intensity[k2]
        x2 = X[k2 + 1]
        y2 = Intensity[k2 + 1]
        b = (y2 - y1) / (x2 - x1)
        c = y2 - b * x2
        xR = (IntensPeak / 2 - c) / b

        FWHM = abs(xR - xL)

        # calculating Sidelobe ratio
        k1 = kc1
        k2 = kc2
        while not (Intensity[k1 - 1] >= Intensity[k1] and Intensity[k1] < Intensity[k1 + 1] and k1 > 2):
            k1 = k1 - 1
        while not (Intensity[k2 - 1] > Intensity[k2] and Intensity[k2] <= Intensity[k2 + 1] and k2 < N - 1):
            k2 = k2 + 1

        SideLobeRatio = max(max(Intensity[0:k1]), max(Intensity[k2 - 1:N])) / IntensPeak

    return FWHM, SideLobeRatio, IntensPeak


def Fun_FitnessLenserror(TargetFWHM, TargetSidlobe, TargetPeakIntensity, TargetFocal_offset, FWHM, SideLobe, IntensPeak,
                         Focal_offset, DOF, Intensity_sum):
    """

    :param TargetFWHM:
    :param TargetSidlobe:
    :param TargetPeakIntensity:
    :param TargetFocal_offset:
    :param FWHM:
    :param SideLobe:
    :param IntensPeak:
    :param Focal_offset:
    :param DOF:
    :param Intensity_sum:
    :return:
    """
    fitness1 = abs(TargetPeakIntensity - IntensPeak) / 100  # 强度偏差500
    fitness2 = abs(Focal_offset - TargetFocal_offset) * 10000  # 焦斑偏移量0.001
    fitness3 = abs(FWHM - TargetFWHM) * 10  # FWHM 0.1
    fitness4 = abs(SideLobe - TargetSidlobe)
    fitness5 = abs(DOF[0] - DOF[1]) + abs(DOF[2] - DOF[1])  # 计算其他波长较深与中心波长较深的偏离
    fitness6 = (abs(Intensity_sum[0] - Intensity_sum[1]) + abs(Intensity_sum[2] - Intensity_sum[1])) / 100
    fitness = fitness1 + fitness2 ** 2 + fitness3 + fitness4 + fitness5 ** 2 * 5 + fitness6
    return fitness


def Fun_GeneRandomGenerationSinglet(N_gene, Nr_gene):
    """

    :param N_gene:
    :param Nr_gene:
    :return:
    """
    Gene_PersonalPresent = N_gene * (np.random.rand(Nr_gene) - 0.5) * 2  # 产生-2pi~2pi的相位跳变；
    Fitness_PersonalPresent = np.inf
    Fitness_PersonalBest = np.inf
    Gene_PersonalBest = Gene_PersonalPresent
    Velocity_PersonalPresent = np.sign(np.random.rand(Nr_gene) - 0.5) * np.random.rand(Nr_gene)
    # print(type(Gene_PersonalPresent), type(Fitness_PersonalPresent), type(Fitness_PersonalBest),
    #       type(Gene_PersonalBest), type(Velocity_PersonalPresent))
    # print(Gene_PersonalPresent.shape, Fitness_PersonalPresent, Fitness_PersonalBest, Gene_PersonalBest.shape,
    #       Velocity_PersonalPresent.shape)
    return Gene_PersonalPresent, Fitness_PersonalPresent, Fitness_PersonalBest, Gene_PersonalBest, Velocity_PersonalPresent


def Fun_GeneRandomGeneration(N_gene, Nr_gene, N_particle, Max_iteration):
    """

    :param N_gene:
    :param Nr_gene:
    :param N_particle:
    :param Max_iteration:
    :return:
    """
    Gene_PersonalPresent = N_gene * (np.random.rand(N_particle, Nr_gene) - 0.5) * 2  # 产生-2pi~2pi的相位跳变；
    Fitness_PersonalPresent = np.zeros((N_particle))
    Fitness_PersonalBest = np.zeros((Max_iteration))
    Gene_PersonalBest = Gene_PersonalPresent
    Velocity_PersonalPresent = np.sign(np.random.rand(N_particle, Nr_gene) - 0.5) * np.random.rand(N_particle, Nr_gene)
    # print(Velocity_PersonalPresent.shape)
    return Gene_PersonalPresent, Fitness_PersonalPresent, Fitness_PersonalBest, Gene_PersonalBest, Velocity_PersonalPresent


def Fun_UpdateParticleSingletBerryPhase(Gene_Lens, Gene_LensPersonalBest, Gene_LensGlobalBest, Gene_LensPersonalBestL,
                                        Velocity, N_gene, N_particle, Nr_gene1):
    """

    :param Gene_Lens:
    :param Gene_LensPersonalBest:
    :param Gene_LensGlobalBest:
    :param Gene_LensPersonalBestL:
    :param Velocity:
    :param N_gene:
    :param N_particle:
    :param Nr_gene1:
    :return:
    """
    # Particle Swarm Parameters
    c1 = 2
    c2 = 2  # 为计算粒子速度的系数 2，2
    w = 0.5  # 计算粒子速度是的权重 0.5
    Vmax = 1
    Vmin = -1
    dt = 0.8
    q = 0.5  # 局部社会因子系数

    for k in range(N_particle):
        # Update velocity
        Velocity[k, :] = w * Velocity[k, :] + c1 * np.random.rand(1, Nr_gene1) * (
                Gene_LensPersonalBest - Gene_Lens[k, :]) + c2 * (
                                 q * np.random.rand(1, Nr_gene1) * (Gene_LensGlobalBest - Gene_Lens[k, :]) + (
                                 1 - q) * np.random.rand(1, Nr_gene1) * (
                                         Gene_LensPersonalBestL - Gene_Lens[k, :]))

        # Update genes
        Gene_Lens[k, :] = Gene_Lens[k, :] + Velocity[k, :] * dt

        # If the gene values reach their boundary, then the values are set as the boundary value, and the sign of velocity is inverted.

        # Lower boundary
        Velocity[k, np.where(Gene_Lens[k, :] <= -N_gene)[0]] = np.absolute(
            Velocity[k, np.where(Gene_Lens[k, :] <= -N_gene)[0]])
        Gene_Lens[k, np.where(Gene_Lens[k, :] <= -N_gene)[0]] = N_gene
        Velocity[k, np.where(Velocity[k, :] < Vmin)[0]] = Vmin

        # Upper boundary velocity
        Velocity[k, np.where(Velocity[k, :] > Vmax)[0]] = Vmax

        # Upper boundary gene of Lens #1 and Lens#2
        Gene_Lens[k, np.where(Gene_Lens[k, :] >= N_gene)[0]] = N_gene
        Velocity[k, np.where(Gene_Lens[k, :] >= N_gene)[0]] = -np.absolute(
            Velocity[k, np.where(Gene_Lens[k, :] >= N_gene)[0]])

    return Gene_Lens, Velocity
