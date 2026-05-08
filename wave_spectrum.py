import numpy as np
from matplotlib import pyplot as plt
def jonswap(Hs, Tp, omega, gamma=3.3):
    '''
        Hs: significant wave height
        Tp: peak period
        gamma: peak enhancement factor
        omega: 角频率
        jonswap wave spectrum, refrerence:
        https://www.sciencedirect.com/science/article/pii/S0960148116301446
        omega=2*pi*f, f为频率, omega为角频率,最终结果需要乘以2*pi
        以与论文结果对比验证
    '''
    fp = 2*np.pi/Tp
    sigma = np.where(omega <= fp, 0.07, 0.09)
    alpha = 0.0624 / (0.230 + 0.0336 * gamma - (0.185 / (1.9+gamma)))
    beta = np.exp(-(omega-fp)**2/(2*(sigma**2)*(fp**2)))
    S = alpha*Hs**2*fp**4*omega**(-5)*gamma**beta*np.exp(-1.25*(fp/omega)**4)*2*np.pi
    # plt.plot(omega, S, label='jonswap')
    # plt.legend()
    # plt.show()
    return S

# # wave spectrum
# Hs = 1.25
# Tp = 8.29
# omega = np.linspace(0.1, 2, 40)
# S = jonswap(Hs, Tp, omega)
