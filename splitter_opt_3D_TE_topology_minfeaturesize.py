######## IMPORTS ########
# General purpose imports
import numpy as np
import sys
import os
sys.path.append("D:\\Program\\Lumerical\\v241\\api\\python") #默认windows lumapi路径
sys.path.append(os. path.dirname(__file__)) #当前目录import os
import lumapi
import scipy as sp

# Optimization specific imports                导入lumopt的相关模块
from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization2D, TopologyOptimization3DLayered
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

cur_path = os.path.dirname(os.path.abspath(__file__))

######## RUNS TOPOLOGY OPTIMIZATION OF A 3D STRUCTURE ########
def runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R, min_feature_size, working_dir, beta = 1):             #定义runSim函数，参数为params, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R, min_feature_size, working_dir, beta
    # 先定义拓扑优化的几何层，然后是modematch，然后是scipy优化器（最小化算法）的参数设置，最后是优化所需的几何参数设置
    ######## DEFINE A 3D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization3DLayered(params=params, eps_min=eps_bg, eps_max=eps_wg, x=x_pos, y=y_pos, z=z_pos, filter_R=filter_R, min_feature_size=min_feature_size, beta=beta)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to modematch to the fundamental TE mode
    #向前的模式光源，分光比为0.5
    fom = ModeMatch(monitor_name = 'fom', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom = 0.5)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    #:param max_iter：最大迭代次数；每次迭代可以进行多次优点和梯度评估。
    # :param method：选择的最小化算法字符串。
    # :scaling_factor：标量或与优化参数长度相同的向量；通常用于缩放优化参数，使其大小在 0 到 1 之间。
    # :param pgtol：投影梯度公差参数 “pgtol”（参见 “BFGS ”或 “L-BFGS-G ”文档）。
    # :param ftol：公差参数'ftol'，当 FOM 的变化小于此值时，可以停止优化。
    # :param scale_initial_gradient_to：强制重新调整梯度，使优化参数的变化至少达到此值 默认值为 0，禁止自动缩放。
    # ：penalt_fun：要添加到优度值中的惩罚函数；它必须是一个接收优化参数向量并返回单一值的函数。优化参数并返回一个值的函数。
    #  :param：penalt_jac：惩罚函数的梯度，必须是一个接收优化参数向量的函数，并返回一个相同长度的向量。

    optimizer = ScipyOptimizers(max_iter=60, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4, scale_initial_gradient_to=0.25)

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    #从现有路径中加载模板脚本，并存储在script中
    script = load_from_lsf(os.path.join(cur_path, 'splitter_base_3D_TE_topology.lsf'))

    ## Here, we substitute the size of the optimization region to properly scale the simulation domain   在这里，我们替换优化区域的大小以适当缩放仿真域
    size_x = max(x_pos) - min(x_pos)
    script = script.replace('opt_size_x=3.5e-6','opt_size_x={:1.6g}'.format(size_x))
    
    size_y = max(y_pos) - min(y_pos)
    script = script.replace('opt_size_y=3.5e-6','opt_size_y={:1.6g}'.format(2*size_y))

    ######## SETTING UP THE OPTIMIZER ########

    wavelengths = Wavelengths(start = 1450e-9, stop = 1650e-9, points = 11)
    opt = Optimization(base_script=script, wavelengths = wavelengths, fom=fom, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False)
    opt.continuation_max_iter = 40 #< How many iterations per binarization step (default is 20)

    ######## RUN THE OPTIMIZER ########
    opt.run(working_dir = working_dir)

if __name__ == '__main__':
    #设备的长宽高
    size_x = 3000
    size_y = 1800
    size_z = 220
    #滤波器的半径与最小特征尺寸
    filter_R = 150e-9
    min_feature_size = filter_R
    #波导的有效介电常数与背景介电常数
    eps_wg = 3.48**2                         #< Effective permittivity for a Silicon waveguide with a thickness of 220nm
    eps_bg = 1.44**2                         #< Permittivity of the SiO2 cladding
    #如果输入参数大于2，则将输入参数赋值给size_x和filter_R
    if len(sys.argv) > 2 :
        size_x = int(sys.argv[1])
        size_y = int(sys.argv[2])
        filter_R = int(sys.argv[3])*1e-9
        print(size_x,size_y,filter_R)

    x_points=int(size_x/20)+1
    y_points=int(size_y/20)+1
    z_points=int(size_z/20)+1

    x_pos = np.linspace(-size_x/2*1e-9,size_x/2*1e-9,x_points)
    y_pos = np.linspace(0,size_y*1e-9,y_points)
    z_pos = np.linspace(-size_z/2*1e-9,size_z/2*1e-9,z_points)

    start_from_2d_result = True
    startingBeta   = 1

#        startingParams = None
#        startingParams = 0.5*np.ones((x_points,y_points))   #< Start with the domain filled with (eps_max+eps_min)/2
#        startingParams = np.ones((x_points,y_points))       #< Start with the domain filled with eps_max
    startingParams = np.zeros((x_points,y_points))      #< Start with the domain filled with eps_min
        
    working_dir = os.path.join(cur_path, 'splitter_3D_TE_topo_x{:04d}_y{:04d}_f{:04d}'.format(size_x,size_y,int(filter_R*1e9)))
    runSim(startingParams, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R=filter_R, min_feature_size=min_feature_size, working_dir=working_dir, beta = startingBeta)
