import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn;
import torch.optim as optim
import scipy.integrate as spi
torch.set_default_dtype(torch.float64)  # was set_default_tensor_type (deprecated in torch>=2.1)
class rmftorch(nn.Module):
    def __init__(self,modelpar,rho_p,rho_n,bn_n,bn_p,grid_corer,grid_maxr,device):
        super(rmftorch, self).__init__()
        self.device=device
        #模型系数
        self.nucleonmass=torch.tensor(modelpar["M"])
        self.m_sigma=torch.tensor(modelpar["m_sigma"])
        self.g2=torch.tensor(modelpar["g2"])
        self.g3=torch.tensor(modelpar["g3"])
        self.g_sigma=torch.tensor(modelpar["g_sigma"])
        self.m_omega=torch.tensor(modelpar["m_omega"])
        self.g_omega=torch.tensor(modelpar["g_omega"])
        self.g_rho=torch.tensor(modelpar["g_rho"])
        self.m_rho=torch.tensor(modelpar["m_rho"])
        self.c3=torch.tensor(modelpar["c3"])
        self.electronmass=torch.tensor(modelpar["electronmass"])
        self.echarge=torch.tensor(modelpar["echarge"])
        #初始密度(迭代参数，一维，球对称，球坐标)
        self.rho_n_rato=nn.Parameter(rho_n).to(torch.device(self.device))
        self.rho_p_rato=nn.Parameter(rho_p).to(torch.device(self.device))
        self.bn_n=bn_n
        self.bn_p=bn_p
        
        #网格密度（一维）r从0到maxr 间隔为grid_dr 距离都为百mev
        self.grid_corer=(torch.tensor(grid_corer/1.97).to(torch.device(self.device)) )   #
        self.pn_grid_num=rho_p.size(0)#0.1个fm距离 换算为了百Mev
        self.grid_incore_dr=self.grid_corer/(self.pn_grid_num-1)
        self.grid_maxr=grid_maxr/1.97       #单位是HMev^-1
        self.ele_grid_num=500  #核外电子取点数量
        self.grid_outcore_dr=(self.grid_maxr-self.grid_corer)/(self.ele_grid_num-1)

        #网格信息储存
        self.raduisinsidecore=None
        self.raduisoutsidecore=None
        self.raduis=None
        self.raduis_rightshift=None
        self.raduisdiff=None
        self.raduisdiff=None
        self.raduisdiff_v3=None
        self.raduisdiff_v3=None
        self.updategrid()
        
        #电磁势（一维，每次迭代需要更新）
        self.A_incore=None
        self.A_outcore=None
        self.core_out_Ee=None
        #在updateA后获得的电子密度
        self.electrondensity=None

        #最新的计算的原子核能量
        self.energy_corevalue=None

    def setgridr(self,r):
        self.grid_corer=r
        self.updategrid()
        self.update_electricfield()
        
    def updategrid(self):
        self.grid_incore_dr=(self.grid_corer-0.05)/(self.pn_grid_num-1)
        self.grid_outcore_dr=(self.grid_maxr-self.grid_corer)/(self.ele_grid_num-1)
        self.raduisinsidecore=torch.linspace(start=0.05,end=self.grid_corer.item(),steps=self.pn_grid_num,device=self.device)
        self.raduisoutsidecore=torch.linspace(start=self.grid_corer.item(),end=self.grid_maxr,steps=self.ele_grid_num,device=self.device)
        self.raduis=torch.cat([self.raduisinsidecore,self.raduisoutsidecore[1:]])
        self.raduis_rightshift=torch.roll(self.raduis,1)
        self.raduisdiff=(self.raduis-self.raduis_rightshift)
        self.raduisdiff=self.raduisdiff[1:]
        self.raduisdiff_v3=(torch.pow(self.raduis,3)-torch.pow(self.raduis_rightshift,3))
        self.raduisdiff_v3=self.raduisdiff_v3[1:]

    def rho_p(self):
        protonnum=self.zhengding(self.rho_p_rato) #单位为 e*HMev^3
        # protonnum_rightshift1=torch.roll(protonnum,1)
        # protonchargemiddle=(protonnum+protonnum_rightshift1)/2
        # protonchargemiddle=protonchargemiddle[1:]
        # totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)
        totalprotonnum=self.Volume_intergrate(protonnum)
        rhop=self.bn_p/totalprotonnum*protonnum
        rhop[-1]=0
        return rhop
    
    def rho_n(self):
        neutronnum=(self.zhengding(self.rho_n_rato)) #单位为 e*HMev^3
        # neutronnum_rightshift1=torch.roll(neutronnum,1)
        # neutronchargemiddle=(neutronnum+neutronnum_rightshift1)/2
        # neutronchargemiddle=neutronchargemiddle[1:]
        # totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*neutronchargemiddle)
        totalneutronnum=self.Volume_intergrate(neutronnum)
        rhon=self.bn_n/totalneutronnum*neutronnum
        rhon[-1]=0
        return rhon
    
    def partialsquare_r(self,rho,grid_step):
        #r=-1 -2 处=r=0处，r=rmax+1 +2 =0 此处理仅适合于强作用场，不适合电磁场，电磁场直接求出无须使用此步骤进行迭代
        
        rho_rightshift1=torch.roll(rho,1)
        rho_rightshift1[0]=rho_rightshift1[2]

        rho_rightshift2=torch.roll(rho,2)
        rho_rightshift2[1]=rho_rightshift2[3]
        rho_rightshift2[0]=rho_rightshift2[4]

        rho_leftshift1=torch.roll(rho,-1)
        rho_leftshift1[-1]=0

        rho_leftshift2=torch.roll(rho,-2)
        rho_leftshift2[-2]=0
        rho_leftshift2[-1]=0
        # print(rho_leftshift2,rho_leftshift1,rho,rho_rightshift2,rho_rightshift1)
        partialsquarer=1/(12*grid_step*grid_step)*(-rho_leftshift2 +16*rho_rightshift1-30*rho+16*rho_leftshift1-rho_rightshift2)
        #o2
        # partialsquarer[1]=(rho[0]+rho[2]-2*rho[1])/(grid_step*grid_step)
        
        # partialsquarer[0]=(-rho[3]+4*rho[2]-5*rho[1]+2*rho[0])/grid_step**2
        #o4
        # partialsquarer[0]=(35 * rho[0] - 104 * rho[1] + 114 * rho[2]  - 56 * rho[3] + 11 * rho[4]) / (12 * grid_step**2)
        # partialsquarer[0]=partialsquarer[1]
        partialsquarer[-2]=(rho_leftshift1[-2]+rho_rightshift1[-2]-2*rho[-2])/(grid_step*grid_step)
        partialsquarer[-1]=0

        partialr=1/(12*grid_step)*(rho_rightshift2-8*rho_rightshift1+8*rho_leftshift1- rho_leftshift2)


        
        # partialr[1]=(rho[2]-rho[1])/grid_step/2
        # partialr[0]=(rho[1]-rho[0])/grid_step/2
        # partialr[1]=(-rho[3]+4*rho[2]-3*rho[1])/grid_step/2
        # partialr[0]=(-rho[2]+4*rho[1]-3*rho[0])/grid_step/2
        # partialr[0]=partialr[1]
        # o4
        # partialr[1]=-(1/(12*grid_step)*(25*rho[1]-48*rho[2]+36*rho[3]-16*rho[4]+3*rho[5]))
        # partialr[0]=-(1/(12*grid_step)*(25*rho[0]-48*rho[1]+36*rho[2]-16*rho[3]+3*rho[4]))
        partialr[-1]=0
        partialr[-2]=(rho_leftshift1[-2]-rho_rightshift1[-2])/grid_step/2

        res=2/self.raduisinsidecore*partialr+partialsquarer
        # res[0]=res[2]
        # res[1]=res[2]
        # res[-1]=0
        # res[-2]=0



        # res=res.detach()
        return res
        

    def partialr_square(self,rho,grid_step):
        #r=-1 -2 处=r=0处，r=rmax+1 +2 =0 此处理仅适合于强作用场，不适合电磁场，电磁场直接求出无须使用此步骤进行迭代

        rho_rightshift1=torch.roll(rho,1)
        rho_rightshift1[0]=rho_rightshift1[2]

        rho_rightshift2=torch.roll(rho,2)
        rho_rightshift2[1]=rho_rightshift2[3]
        rho_rightshift2[0]=rho_rightshift2[4]

        rho_leftshift1=torch.roll(rho,-1)
        rho_leftshift1[-1]=0

        rho_leftshift2=torch.roll(rho,-2)
        rho_leftshift2[-2]=0
        rho_leftshift2[-1]=0

        # partialr=1/(12*self.grid_incore_dr)*(rho_leftshift2-8*rho_leftshift1+8*rho_rightshift1-rho_rightshift2)
        #o4
        partialr=1/(12*grid_step)*(rho_rightshift2-8*rho_rightshift1+8*rho_leftshift1- rho_leftshift2)
        #o2
        # partialr[1]=(rho[2]-rho[1])/grid_step/2
        # partialr[0]=(rho[1]-rho[0])/grid_step/2
        # partialr[1]=(-rho[3]+4*rho[2]-3*rho[1])/grid_step/2
        # partialr[0]=(-rho[2]+4*rho[1]-3*rho[0])/grid_step/2
        #o4
        # partialr[1]=-(1/(12*grid_step)*(25*rho[1]-48*rho[2]+36*rho[3]-16*rho[4]+3*rho[5]))
        # partialr[0]=-(1/(12*grid_step)*(25*rho[0]-48*rho[1]+36*rho[2]-16*rho[3]+3*rho[4]))
        # partialr[0]=partialr[1]
        partialr[-1]=0
        partialr[-2]=(rho_leftshift1[-2]-rho_rightshift1[-2])/grid_step/2
        # partialr=partialr.detach()
        return partialr*partialr
    def Volume_intergrate(self,rho):
        core_rho_tobeintergrated=self.raduis[:self.pn_grid_num]**2*rho
        res=torch.sum((2*core_rho_tobeintergrated[1:]-torch.diff(core_rho_tobeintergrated))/2*self.raduisdiff[:self.pn_grid_num-1]*torch.pi*4)
        res+=self.raduisinsidecore[0]**3*torch.pi*4*rho[0]/6
        return res
    def findsigma(self,startpoint=0.0,step=0.1,accuracy=4):
        #在迭代的时候可能会自动梯度下降到小于0的状态，所以要进行检查一下
        rho_p=self.rho_p()
        rho_n=self.rho_n()

        #设置精度：
        pointnum=rho_p.size()

        deltasigma=10000
        sigma=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        partialsigma=(torch.zeros(pointnum)).to(torch.device(self.device))
        cout=0
        while torch.sum(abs(deltasigma)>abs(sigma)*torch.pow(torch.tensor(0.1),accuracy))>0:

            m_n=self.nucleonmass+self.g_sigma*sigma
            t_np=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_p,1/3)/m_n
            t_nn=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_n,1/3)/m_n
            rho_ps=torch.pow(m_n,3)/torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*(0.5*(t_np*torch.pow(1+torch.pow(t_np,2),0.5)-torch.arcsinh(t_np)))
            rho_ns=torch.pow(m_n,3)/torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*(0.5*(t_nn*torch.pow(1+torch.pow(t_nn,2),0.5)-torch.arcsinh(t_nn)))
            newsigma=-(self.g2*torch.pow(sigma,2)+self.g3*torch.pow(sigma,3)+self.g_sigma*(rho_ns+rho_ps)-partialsigma)/torch.pow(self.m_sigma,2)
            deltasigma=newsigma-sigma
            sigma=sigma+deltasigma*step
            newpartialsigma=self.partialsquare_r(sigma,self.grid_incore_dr)
            partialsigma=partialsigma+(newpartialsigma-partialsigma)*step
            # partialsigma=newpartialsigma
            cout=cout+1

        # print('sigma:cout=',cout,sigma)
        
        return sigma,partialsigma
    def findomega(self,startpoint=0.0,step=0.1,accuracy=4):
        #在迭代的时候可能会自动梯度下降到小于0的状态，所以要进行检查一下
        rho_p=self.rho_p()
        rho_n=self.rho_n()
        #设置精度：
        pointnum=rho_p.size()

        deltaomega=10000
        omega=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        partialomega=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        cout=0
        while torch.sum(((abs(deltaomega))>abs(omega)*torch.pow(torch.tensor(0.1),accuracy)))>0:

            omeganew=(self.g_omega*(rho_p+rho_n)+partialomega-self.c3*omega**3)/torch.pow(self.m_omega,2)
            deltaomega=omeganew-omega
            omega=omega+deltaomega*step
            newpartialomega=self.partialsquare_r(omega,self.grid_incore_dr)
            partialomega=partialomega+(newpartialomega-partialomega)*step
            # partialomega=newpartialomega
            cout=cout+1
            # print(omeganew)
        # print('omega:cout=',cout,omega)
        return omega,partialomega
    def findrho(self,startpoint=0.0,step=0.01,accuracy=4):
        #在迭代的时候可能会自动梯度下降到小于0的状态，所以要进行检查一下
        rho_p=self.rho_p()
        rho_n=self.rho_n()
        #设置精度：
        pointnum=rho_p.size()

        deltarho=10000
        rho=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        partialrho=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        cout=0
        while torch.sum((abs(deltarho)>abs(rho)*torch.pow(torch.tensor(0.1),accuracy)))>0:
            

            rhonew=(self.g_rho*(rho_p-rho_n)+partialrho)/torch.pow(self.m_rho,2)
            deltarho=rhonew-rho
            rho=rho+deltarho*step
            newpartialrho=self.partialsquare_r(rho,self.grid_incore_dr)
            partialrho=partialrho+(newpartialrho-partialrho)*step
            # partialrho=newpartialrho
            cout=cout+1
            
        # print('rho:cout=',cout)
        return rho,partialrho

    def update_electricfield(self):
        self.updategrid()
        protoncharge=self.rho_p() #单位为 e*HMev^3
        protoncharge_rightshift1=torch.roll(protoncharge,1)
        protonchargemiddle=(protoncharge+protoncharge_rightshift1)/2
        protonchargemiddle=protonchargemiddle[1:]

        totalprotoncharge=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)
        core_out_v=(self.grid_maxr**3-self.grid_corer**3)*4/3*torch.pi#核外体积
        
        electrondensity=totalprotoncharge/core_out_v#电子密度
        self.electrondensity=electrondensity
        # totalgridnum=self.ele_grid_num+self.rho_p.size(0)
        totaldensitydis=torch.cat([protonchargemiddle,-torch.ones(self.ele_grid_num-1,device=self.device)*electrondensity],dim=0)
        elecharge=totaldensitydis*self.raduisdiff_v3*4*torch.pi/3#单位是e
        eleforce=torch.zeros(elecharge.size(),device=self.device)
        # raduis=self.raduis[1:]
        raduis=self.raduis
        for i in range(elecharge.size(0)):
            eleforce[i]=torch.sum(elecharge[:i+1])/(4*torch.pi*raduis[i]**2) #单位是e*HMev^2

        
        self.core_out_Ee=torch.sum(torch.cat([0.5*(eleforce[self.pn_grid_num-1:]*self.echarge)**2])*self.raduisdiff_v3[self.pn_grid_num-1:]*4*torch.pi/3)
        # tempcoreoute=spi.simps(y=(eleforce[self.pn_grid_num-1:].detach().cpu().numpy()*0.303)**2*0.5*4*np.pi*raduis[self.pn_grid_num:].detach().cpu().numpy()**2,x=raduis[self.pn_grid_num:].detach().cpu().numpy())

        elepdiff=(eleforce*self.raduisdiff)#单位是e*HMeV
        # corep_surface=elep[self.pn_grid_num-1:]#gridnum-1是因为段数比点数少1
        elep=torch.zeros(totaldensitydis.size(0),device=self.device)
        for i in range(totaldensitydis.size(0)):
            elep[i]=torch.sum(elepdiff[i:])

        # elep=torch.zeros(totalgridnum,device=self.device)
        # for i in range(totalgridnum):
        #     elep[i]=torch.sum(elepdiff[i:])
        self.A_incore=elep[:self.pn_grid_num]*self.echarge#单位是HMeV  ******
        self.A_outcore=torch.cat([elep[self.pn_grid_num-1:]*self.echarge,torch.zeros(1,device=self.device)])
        ###------------debug--------------###
        # print("self.grid_maxr",self.grid_maxr)
        # print("self.grid_corer",self.grid_corer)
        # print("raduis:",raduis)
        # print('raduisinsidecore:',self.raduisinsidecore)
        # print('raduisdiff_v3:',torch.sum(self.raduisdiff_v3[self.pn_grid_num-1:]*4*torch.pi/3))
        # print('core_out_v:',core_out_v)
        # print((eleforce*self.echarge))
        # print("protoncharge:",protoncharge)
        # print("electrondensity:",electrondensity)
        # print('totalelectroncharge:',torch.sum(elecharge[self.pn_grid_num-1:]).detach().cpu().numpy())
        # print('totalprotoncharge:',totalprotoncharge.detach().cpu().numpy(),torch.sum(elecharge[:self.pn_grid_num-1]).detach().cpu().numpy())
        # print('totalcharge:',torch.sum(elecharge))
        # print("elecharge:",elecharge)
        # print("totaldensitydis:",totaldensitydis)
        # print("raduisdiff_v3*4*torch.pi/3:",self.raduisdiff_v3*4*torch.pi/3)
        # print("eleforce:",eleforce)
        # print("tempcoreoute",tempcoreoute)
        # print("core_out_Ee",self.core_out_Ee)
        
    def eps_core(self,step=0.1):
        rho_p=self.rho_p()
        rho_n=self.rho_n()
        self.update_electricfield()
        sigma,partial2sigma=self.findsigma(step=step)
        omega,partial2omega=self.findomega(step=step)
        rho,partial2rho=self.findrho(step=step)
        # rho=torch.zeros(rho.size(),device=self.device)
        m_n=self.nucleonmass+self.g_sigma*sigma
        m_e=self.electronmass
        t_np=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_p,1/3)/m_n
        t_nn=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_n,1/3)/m_n
        a1=(t_np*torch.pow(torch.pow(t_np,2)+1,0.5)*(1+2*torch.pow(t_np,2))-torch.arcsinh(t_np))
        a2=(t_nn*torch.pow(torch.pow(t_nn,2)+1,0.5)*(1+2*torch.pow(t_nn,2))-torch.arcsinh(t_nn))
        partialsigma=self.partialr_square(sigma,self.grid_incore_dr)
        partialomega=self.partialr_square(omega,self.grid_incore_dr)
        partialrho=self.partialr_square(rho,self.grid_incore_dr)
        partialA=self.partialr_square(self.A_incore,self.grid_incore_dr)
        # eps=torch.pow(m_n,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*(a1+a2)+self.g_omega*omega*(rho_n+rho_p)+self.g_rho*rho*(rho_p-rho_n)+0.5*torch.pow(self.m_sigma,2)*torch.pow(sigma,2)+1/3*self.g2*torch.pow(sigma,3)+0.25*self.g3*torch.pow(sigma,4)-0.5*torch.pow(self.m_omega,2)*torch.pow(omega,2)-0.5*torch.pow(self.m_rho*rho,2)+0.5*(partialsigma+partialomega+partialrho+partialA)
        eps=torch.pow(m_n,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*(a1+a2)\
        +0.5*torch.pow(self.m_omega,2)*torch.pow(omega,2)+0.75*self.c3*torch.pow(omega,4)\
        +0.5*torch.pow(self.m_rho*rho,2)\
        +0.5*torch.pow(self.m_sigma,2)*torch.pow(sigma,2)+1/3*self.g2*torch.pow(sigma,3)+0.25*self.g3*torch.pow(sigma,4)\
        +0.5*(partialsigma+partialomega+partialrho+partialA)

        #--------debug------------#
        # print('eps:',eps)
        # print('partialsigma:',partialsigma)
        # print('partialomega:',partialomega)
        # print('partialrho:',partialrho)
        # print('partialA:',partialA)
        return eps

        
    def energy_core(self,step=0.1):
        # print("energycoreV:",self.raduisdiff_v3[:self.pn_grid_num]*4*torch.pi/3)
        core_eps=self.eps_core(step=step)
        self.energy_corevalue=self.Volume_intergrate(core_eps)
        # core_eps_rightshift1=torch.roll(core_eps,1)
        # core_epsmiddle=(core_eps+core_eps_rightshift1)/2
        # core_epsmiddle=core_epsmiddle[1:]
        # self.energy_corevalue=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*4*torch.pi/3*core_epsmiddle)
        return self.energy_corevalue
    
    def getmaxrho(self):
        corev=((self.grid_corer**3)*4*torch.pi/3)
        return ((self.bn_n+self.bn_p)/corev)/1.97**3 #单位为 fm-3
    
    def energy_A_outcore(self):
        e=self.echarge
        R0=self.grid_maxr
        r0=self.grid_corer
        Z=self.bn_p
        pi=torch.pi
        ne=Z/((R0**3-r0**3)*4/3*np.pi)
        ff=(-((e**2 *(16* ne**2* pi**2*r0*(5*r0**6 - 9 *r0**5* R0 + 5 *r0**3 *R0**3 - R0**6) + \
            60 *ne* pi *r0 *(r0 - R0)**2* (2* r0 + R0) *Z + 45 *(r0 - R0) *Z**2))/(\
            360* pi *r0 *R0)))
        # print(Z,R0,r0,ne)
        return ff
    def energy_electron(self):
        partialA=self.partialr_square(self.A_outcore,self.grid_outcore_dr)
        electrondensity=self.electrondensity#电子密度
        t_ne=torch.pow(3*torch.pi**2*electrondensity,1/3)/self.electronmass
        ae=(t_ne*torch.pow(torch.pow(t_ne,2)+1,0.5)*(1+2*torch.pow(t_ne,2))-torch.arcsinh(t_ne))
        eps_electron=torch.pow(self.electronmass,4)/(8*torch.pi**2)*ae
        # energy_ele=torch.sum((self.raduisdiff_v3[self.pn_grid_num-1:])*4*torch.pi/3*(eps_electron+partialA*0.5))
        # energy_ele=torch.sum((self.raduisdiff_v3[self.pn_grid_num-1:])*4*torch.pi/3*(partialA[:-1]*0.5))
        energy_ele=torch.sum((self.raduisdiff_v3[self.pn_grid_num-1:])*4*torch.pi/3*(eps_electron))+self.energy_A_outcore()
        # print("outcoreAenergy:",self.energy_A_outcore())
        return energy_ele
    def new_energy_density(self,rho,radius,ener_core):
        #不保留梯度信息
        with torch.no_grad():
            tmp=self.energy_corevalue
            self.energy_corevalue=ener_core
            tmpr=self.grid_corer
            self.setgridr(radius)
            eee=self.energy_density(rho)
            self.energy_corevalue=tmp
            self.grid_corer=tmpr
            self.updategrid()
            self.update_electricfield()
        return eee
    def energy_density(self,rho,step=0.1):
        # rho_p=self.zhengding(self.rho_p())
        # protonnum=rho_p #单位为 e*HMev^3
        # protonnum_rightshift1=torch.roll(protonnum,1)
        # protonnummiddle=(protonnum+protonnum_rightshift1)/2
        # protonnummiddle=protonnummiddle[1:]
        # totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonnummiddle)
        # totalelenum=totalprotonnum
        # totalelemass=totalelenum*self.electronmass
        corev=((self.grid_corer**3)*4*torch.pi/3)
        energy_density=torch.zeros(rho.size())
        atom_num_density=torch.zeros(rho.size())
        gridmaxr=self.grid_maxr
        for i in range(rho.size(0)):
            targetedV=(self.bn_n+self.bn_p)/rho[i]
            atom_num_density[i]=(1/targetedV)/1.97**3# fm-3
            targetedradius_atom=(targetedV*3/4/torch.pi)**(1/3)
            self.grid_maxr=targetedradius_atom
            self.updategrid()
            self.update_electricfield()
            elev=targetedV-corev
            if elev>0:
                # partialA=self.partialr_square(self.A_outcore,self.grid_outcore_dr)
                # core_out_v=elev#核外体积
                # electrondensity=totalprotonnum/core_out_v#电子密度
                # t_ne=torch.pow(3*torch.pi**2*electrondensity,1/3)/self.electronmass
                # ae=(t_ne*torch.pow(torch.pow(t_ne,2)+1,0.5)*(1+2*torch.pow(t_ne,2))-torch.arcsinh(t_ne))
                # eps_electron=torch.pow(self.electronmass,4)/(8*torch.pi**2)*ae
                # energy_ele=torch.sum((self.raduisdiff_v3[self.pn_grid_num-1:])*4*torch.pi/3*(eps_electron+partialA*0.5))

                # print(partialA)
                energy_density[i]=(self.energy_corevalue+self.energy_electron())/targetedV
                # energy_density[i]=(self.energy_electron())
                # print("energy_electron",self.energy_electron())
                # print("targetedradius_atom",targetedradius_atom)
            else:
                energy_density[i]=0
        #还原系数
        self.grid_maxr=gridmaxr
        self.updategrid()
        return energy_density,atom_num_density
            
        
    def baryons(self):
        protonnum=(self.rho_p()) #单位为 e*HMev^3
        protonnum_rightshift1=torch.roll(protonnum,1)
        protonchargemiddle=(protonnum+protonnum_rightshift1)/2
        protonchargemiddle=protonchargemiddle[1:]
        totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)

        neutronnum=(self.rho_n()) #单位为 e*HMev^3
        neutronnum_rightshift1=torch.roll(neutronnum,1)
        neutronmiddle=(neutronnum+neutronnum_rightshift1)/2
        neutronmiddle=neutronmiddle[1:]
        totalneutronnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*neutronmiddle)

        return totalneutronnum,totalprotonnum

    def chemicalpotentials(self,omega,sigma,rho):
        rho_p=self.rho_p()
        rho_n=self.rho_n()
        
        cp_p=self.g_omega*omega+self.g_rho*rho+self.echarge*self.A_incore+torch.pow(torch.pow(3*torch.pi*torch.pi*rho_p,2/3)+torch.pow(self.nucleonmass+self.g_sigma*sigma,2),0.5)
        cp_n=self.g_omega*omega-self.g_rho*rho+torch.pow(torch.pow(3*torch.pi*torch.pi*rho_n,2/3)+torch.pow(self.nucleonmass+self.g_sigma*sigma,2),0.5)
        return cp_n*rho_n+cp_p*rho_p
    def constancycondition(self,cp):
        return torch.sum(torch.abs(self.partial_periodicity(cp,self.partialstep)))
    def zhengding(self,a):
        b=a.clone()
        # for i in range(a.size(0)):
        #             if a[i]<0:
        #                 b[i]=0.0001
        #             else:
        #                 b[i]=a[i]
        b[-1]=0
        return b**2
    