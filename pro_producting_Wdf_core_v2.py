import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn;
import torch.optim as optim
import scipy.integrate as spi
torch.set_default_dtype(torch.float64)
class rmftorch(nn.Module):
    def __init__(self,modelpar,rho_p,rho_n,bn_n,bn_p,grid_corer,grid_maxr,device):
        super(rmftorch, self).__init__()
        self.device=device
        #模型系数
        self.neutronmass=torch.tensor(modelpar["M_n"])
        self.protonmass=torch.tensor(modelpar["M_p"])
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

        #一些帮助处理的参数
        self.rightupperones=torch.triu(torch.ones(self.pn_grid_num-1,self.pn_grid_num-1,device=self.device), diagonal=0)
        self.leftlowones=torch.tril(torch.ones(self.pn_grid_num-1,self.pn_grid_num-1,device=self.device), diagonal=0)
        self.leftlowones_allspace=torch.tril(torch.ones(self.pn_grid_num+self.ele_grid_num-2,self.pn_grid_num+self.ele_grid_num-2,device=self.device), diagonal=0)
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
        
        # self.grid_incore_dr=(self.grid_corer-0.01)/(self.pn_grid_num-1)
        # self.raduisinsidecore=torch.linspace(start= 0.01,end=self.grid_corer.item(),steps=self.pn_grid_num,device=self.device)

        self.grid_outcore_dr=(self.grid_maxr-self.grid_corer)/(self.ele_grid_num-1)

        
        self.grid_incore_dr=(self.grid_corer)/(self.pn_grid_num)
        self.raduisinsidecore=torch.linspace(start= self.grid_incore_dr,end=self.grid_corer.item(),steps=self.pn_grid_num,device=self.device)
        
        self.raduisoutsidecore=torch.linspace(start=self.grid_corer.item(),end=self.grid_maxr,steps=self.ele_grid_num,device=self.device)
        self.raduis=torch.cat([self.raduisinsidecore,self.raduisoutsidecore[1:]])
        self.raduis_rightshift=torch.roll(self.raduis,1)
        self.raduisdiff=(self.raduis-self.raduis_rightshift)
        self.raduisdiff=self.raduisdiff[1:]
        self.raduisdiff_v3=(torch.pow(self.raduis,3)-torch.pow(self.raduis_rightshift,3))
        self.raduisdiff_v3=self.raduisdiff_v3[1:]

    def rho_p(self):
        # self.rho_p_rato[0]=2*self.rho_p_rato[1]-self.rho_p_rato[2]
        protonnum=self.zhengding(self.rho_p_rato) #单位为 e*HMev^3
        # protonnum_rightshift1=torch.roll(protonnum,1)
        # protonchargemiddle=(protonnum+protonnum_rightshift1)/2
        # protonchargemiddle=protonchargemiddle[1:]
        totalprotonnum=torch.sum(self.Volume_intergrate(protonnum))
        # protonchargemiddle[0]=protonnum[0]
        # v3=torch.concatenate([torch.tensor([self.raduisinsidecore[0]**3],device=self.device),self.raduisdiff_v3[:self.pn_grid_num-1]])
        # totalprotonnum=torch.sum(v3*torch.pi*4/3*protonchargemiddle)
        # totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)
        # totalprotonnum+=self.raduisinsidecore[0]**3*torch.pi*4/3*protonnum[0]/1.4
        rhop=self.bn_p/totalprotonnum*protonnum
        rhop[-1]=0
        # rhop[0]=2*rhop[1]-rhop[2]
        return rhop
    
    def rho_n(self):
        # self.rho_n_rato[0]=2*self.rho_n_rato[1]-self.rho_n_rato[2]
        neutronnum=(self.zhengding(self.rho_n_rato)) #单位为 e*HMev^3
        # neutronnum_rightshift1=torch.roll(neutronnum,1)
        # neutronchargemiddle=(neutronnum+neutronnum_rightshift1)/2
        # neutronchargemiddle=neutronchargemiddle[1:]
        # neutronchargemiddle[0]=neutronnum[0]
        # v3=torch.concatenate([torch.tensor([self.raduisinsidecore[0]**3],device=self.device),self.raduisdiff_v3[:self.pn_grid_num-1]])
        # totalprotonnum=torch.sum(v3*torch.pi*4/3*neutronchargemiddle)
        totalneutronnum=torch.sum(self.Volume_intergrate(neutronnum))
        # totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*neutronchargemiddle)
        # totalprotonnum+=self.raduisinsidecore[0]**3*torch.pi*4/3*neutronnum[0]/1.4
        rhon=self.bn_n/totalneutronnum*neutronnum
        rhon[-1]=0
        # rhon[0]=2*rhon[1]-rhon[2]
        return rhon
    
    def partialsquare_r(self,rho,grid_step):
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

        partialsquarer[-2]=(rho_leftshift1[-2]+rho_rightshift1[-2]-2*rho[-2])/(grid_step*grid_step)
        partialsquarer[-1]=0

        partialr=1/(12*grid_step)*(rho_rightshift2-8*rho_rightshift1+8*rho_leftshift1- rho_leftshift2)

        partialr[-1]=0
        partialr[-2]=(rho_leftshift1[-2]-rho_rightshift1[-2])/grid_step/2
        partialr[1]=-(1/(12*grid_step)*(25*rho[1]-48*rho[2]+36*rho[3]-16*rho[4]+3*rho[5]))
        partialr[0]=-(1/(12*grid_step)*(25*rho[0]-48*rho[1]+36*rho[2]-16*rho[3]+3*rho[4]))#不然会降为0不好
        res=2/self.raduisinsidecore*partialr+partialsquarer

        return res
        

    def partialr_square(self,rho,grid_step):

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

        partialr=1/(12*grid_step)*(rho_rightshift2-8*rho_rightshift1+8*rho_leftshift1- rho_leftshift2)
        partialr[1]=-(1/(12*grid_step)*(25*rho[1]-48*rho[2]+36*rho[3]-16*rho[4]+3*rho[5]))
        partialr[0]=-(1/(12*grid_step)*(25*rho[0]-48*rho[1]+36*rho[2]-16*rho[3]+3*rho[4]))
        partialr[-1]=0
        partialr[-2]=(rho_leftshift1[-2]-rho_rightshift1[-2])/grid_step/2
        return partialr*partialr
    # def doubleintergrate_incore(self,partial2rho):
    #     grid_step=self.grid_incore_dr
    #     partial2rhor=self.raduisinsidecore.to(torch.float64)**2*partial2rho
    #     partial2rhorm=(2*partial2rhor[1:]-torch.diff(partial2rhor))/2
    #     partialrhor2=torch.zeros_like(partial2rho)
    #     for i in range(partial2rhorm.size(0)):
    #         partialrhor2[i]=torch.sum(partial2rhorm[:i+1])*grid_step

    #     partialrho=partialrhor2/self.raduisinsidecore**2
    #     partialrhom=(2*partialrho[1:]-torch.diff(partialrho))/2
    #     rho=torch.zeros_like(partialrho)
    #     for i in range(partialrho.size(0)):
    #         rho[i]=-torch.sum(partialrhom[i:])*grid_step
    #     return rho
    # def singleintergrate_incore(self,partial2rho):
    #     grid_step=self.grid_incore_dr
    #     partial2rhor=self.raduisinsidecore.to(torch.float64)**2*partial2rho
    #     partial2rhorm=(2*partial2rhor[1:]-torch.diff(partial2rhor))/2
    #     partialrhor2=torch.zeros_like(partial2rho)
    #     for i in range(partial2rhorm.size(0)):
    #         partialrhor2[i]=torch.sum(partial2rhorm[:i+1])*grid_step

    #     partialrho=partialrhor2/self.raduisinsidecore**2
    #     return partialrho
    def singleintergrate_incore(self,partial2rho):
        partial2rhor=self.raduisinsidecore**2*partial2rho
        partial2rhorm=(2*partial2rhor[1:]-torch.diff(partial2rhor))/2
        partialrhor2=torch.zeros_like(partial2rho)
        partialrhor2[1:]=torch.sum(partial2rhorm*self.leftlowones,dim=1)*self.grid_incore_dr
        partialrhor2[0]=partial2rhor[0]*self.grid_incore_dr/2
        partialrho=partialrhor2/self.raduisinsidecore**2
        return partialrho
    def doubleintergrate_incore(self,partial2rho):
        # partial2rhor=self.raduisinsidecore**2*partial2rho
        # partial2rhorm=(2*partial2rhor[1:]-torch.diff(partial2rhor))/2
        # partialrhor2=torch.zeros_like(partial2rho)
        # partialrhor2[1:]=torch.sum(partial2rhorm*self.leftlowones,dim=1)*self.grid_incore_dr
        # partialrhor2[0]=partial2rhor[0]*self.grid_incore_dr/2
        # partialrho=partialrhor2/self.raduisinsidecore**2
        partialrho=self.singleintergrate_incore(partial2rho)
        partialrhom=(2*partialrho[1:]-torch.diff(partialrho))/2
        rho=torch.zeros_like(partialrho)
        rho[:-1]=-torch.sum(partialrhom*self.rightupperones,dim=1)*self.grid_incore_dr
        return rho
    def Volume_intergrate(self,rho):
        rhonum=len(rho)
        core_rho_tobeintergrated=self.raduis[:rhonum]**2*rho
        res=(2*core_rho_tobeintergrated[1:]-torch.diff(core_rho_tobeintergrated))/2*self.raduisdiff[:rhonum-1]*torch.pi*4
        # res+=self.raduisinsidecore[0]**3*torch.pi*4*rho[0]/6
        return res

    def findsigma(self,startpoint=0.0,step=0.1,accuracy=4):
        #在迭代的时候可能会自动梯度下降到小于0的状态，所以要进行检查一下
        rho_p=self.rho_p()
        rho_n=self.rho_n()

        #设置精度：
        pointnum=rho_p.size()

        deltasigma=10000
        sigma=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        cout=0
        while torch.sum(abs(deltasigma)>abs(sigma)*torch.pow(torch.tensor(0.1),accuracy))>0:

            m_p=self.protonmass+self.g_sigma*sigma
            m_n=self.neutronmass+self.g_sigma*sigma
            t_np=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_p,1/3)/m_p
            t_nn=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_n,1/3)/m_n
            rho_ps=torch.pow(m_p,3)/torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*(0.5*(t_np*torch.pow(1+torch.pow(t_np,2),0.5)-torch.arcsinh(t_np)))
            rho_ns=torch.pow(m_n,3)/torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*(0.5*(t_nn*torch.pow(1+torch.pow(t_nn,2),0.5)-torch.arcsinh(t_nn)))
            
            newpartialsigma=sigma*torch.pow(self.m_sigma,2)+self.g2*torch.pow(sigma,2)+self.g3*torch.pow(sigma,3)+self.g_sigma*(rho_ns+rho_ps)
            intergratedsigma=self.doubleintergrate_incore(newpartialsigma)
            
            deltasigma=(intergratedsigma-sigma)
            steprand=step*torch.exp(torch.randn_like(deltasigma)*0.1)
            sigma=sigma+deltasigma*steprand
            cout=cout+1
            if cout>1000:
                break        
        return sigma,newpartialsigma
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

            partialomega=omega*torch.pow(self.m_omega,2)-(self.g_omega*(rho_p+rho_n)-self.c3*omega**3)
            intergratedomega=self.doubleintergrate_incore(partialomega)
            deltaomega=intergratedomega-omega
            steprand=step*torch.exp(torch.randn_like(deltaomega)*0.1)
            omega=omega+deltaomega*steprand

            cout=cout+1
            if cout>800:
                break
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
            

            partialrho=rho*torch.pow(self.m_rho,2)-(self.g_rho*(rho_p-rho_n))
            intergratedrho=self.doubleintergrate_incore(partialrho)
            deltarho=intergratedrho-rho
            steprand=step*torch.exp(torch.randn_like(deltarho)*0.1)
            rho=rho+deltarho*steprand
            cout=cout+1
            if cout>800:
                break

            
        # print('rho:cout=',cout)
        return rho,partialrho

    def update_electricfield(self):
        self.updategrid()
        protoncharge=self.rho_p() #单位为 e*HMev^3
        # protoncharge_rightshift1=torch.roll(protoncharge,1)
        # protonchargemiddle=(protoncharge+protoncharge_rightshift1)/2
        # protonchargemiddle=protonchargemiddle[1:]

        # totalprotoncharge=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)
        totalprotoncharge=torch.sum(self.Volume_intergrate(protoncharge))
        core_out_v=(self.grid_maxr**3-self.grid_corer**3)*4/3*torch.pi#核外体积
        
        electrondensity=totalprotoncharge/core_out_v#电子密度
        self.electrondensity=electrondensity
        # totalgridnum=self.ele_grid_num+self.rho_p.size(0)
        totaldensitydis=torch.cat([protoncharge,-torch.ones(self.ele_grid_num-1,device=self.device)*electrondensity],dim=0)
        # elecharge=totaldensitydis*self.raduisdiff_v3*4*torch.pi/3#单位是e
        elecharge=self.Volume_intergrate(totaldensitydis)
        eleforce=torch.zeros_like(totaldensitydis)
        # print(torch.sum(self.leftlowones_allspace*elecharge,dim=1).shape)
        # print(self.raduis[1:].shape)
        eleforce[1:]=torch.sum(self.leftlowones_allspace*elecharge,dim=1)/(4*torch.pi*self.raduis[1:]**2)
        
        # eleforce=torch.zeros(elecharge.size()+1,device=self.device)
        # raduis=self.raduis[1:]
        # raduis=self.raduis
        # for i in range(elecharge.size(0)):
        #     eleforce[i]=torch.sum(elecharge[:i+1])/(4*torch.pi*raduis[i+1]**2) #单位是e*HMev^2
        self.core_in_Ee_density=0.5*(eleforce[:self.pn_grid_num]*self.echarge)**2
        # self.core
        # self.core_out_Ee=torch.sum([0.5*(eleforce[self.pn_grid_num-1:]*self.echarge)**2]*self.raduisdiff_v3[self.pn_grid_num-1:]*4*torch.pi/3)
        # self.core_in_Ee_density=0.5*(eleforce[:self.pn_grid_num]*self.echarge)**2
        # tempcoreoute=spi.simps(y=(eleforce[self.pn_grid_num-1:].detach().cpu().numpy()*0.303)**2*0.5*4*np.pi*raduis[self.pn_grid_num:].detach().cpu().numpy()**2,x=raduis[self.pn_grid_num:].detach().cpu().numpy())

        # elepdiff=(eleforce*self.raduisdiff)#单位是e*HMeV
        # corep_surface=elep[self.pn_grid_num-1:]#gridnum-1是因为段数比点数少1
        # elep=torch.zeros(totaldensitydis.size(0),device=self.device)
        # for i in range(totaldensitydis.size(0)):
            # elep[i]=torch.sum(elepdiff[i:])

        # elep=torch.zeros(totalgridnum,device=self.device)
        # for i in range(totalgridnum):
        #     elep[i]=torch.sum(elepdiff[i:])
        # self.A_incore=elep[:self.pn_grid_num]*self.echarge#单位是HMeV  ******
        # self.A_outcore=torch.cat([elep[self.pn_grid_num-1:]*self.echarge,torch.zeros(1,device=self.device)])
        ##------------debug--------------###
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
        m_p=self.protonmass+self.g_sigma*sigma
        m_n=self.neutronmass+self.g_sigma*sigma
        m_e=self.electronmass
        t_np=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_p,1/3)/m_p
        t_nn=torch.pow(3*torch.pow(torch.tensor(torch.pi, requires_grad=True),2)*rho_n,1/3)/m_n
        a1=(t_np*torch.pow(torch.pow(t_np,2)+1,0.5)*(1+2*torch.pow(t_np,2))-torch.arcsinh(t_np))
        a2=(t_nn*torch.pow(torch.pow(t_nn,2)+1,0.5)*(1+2*torch.pow(t_nn,2))-torch.arcsinh(t_nn))
        # partialsigma=self.partialr_square(sigma,self.grid_incore_dr)
        # partialomega=self.partialr_square(omega,self.grid_incore_dr)
        # partialrho=self.partialr_square(rho,self.grid_incore_dr)
        # partialA=self.partialr_square(self.A_incore,self.grid_incore_dr)
        partialsigma=self.singleintergrate_incore(partial2sigma)**2
        partialomega=self.singleintergrate_incore(partial2omega)**2
        partialrho=self.singleintergrate_incore(partial2rho)**2

        # eps=torch.pow(m_n,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*(a1+a2)+self.g_omega*omega*(rho_n+rho_p)+self.g_rho*rho*(rho_p-rho_n)+0.5*torch.pow(self.m_sigma,2)*torch.pow(sigma,2)+1/3*self.g2*torch.pow(sigma,3)+0.25*self.g3*torch.pow(sigma,4)-0.5*torch.pow(self.m_omega,2)*torch.pow(omega,2)-0.5*torch.pow(self.m_rho*rho,2)+0.5*(partialsigma+partialomega+partialrho+partialA)
        # eps=torch.pow(m_n,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*(a1+a2)\
        eps=torch.pow(m_p,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*a1\
        +torch.pow(m_n,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*a2\
        +0.5*torch.pow(self.m_omega,2)*torch.pow(omega,2)+0.75*self.c3*torch.pow(omega,4)\
        +0.5*torch.pow(self.m_rho*rho,2)\
        +0.5*torch.pow(self.m_sigma,2)*torch.pow(sigma,2)+1/3*self.g2*torch.pow(sigma,3)+0.25*self.g3*torch.pow(sigma,4)\
        +0.5*(partialsigma+partialomega+partialrho)+self.core_in_Ee_density

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
        # core_eps_rightshift1=torch.roll(core_eps,1)
        # core_epsmiddle=(core_eps+core_eps_rightshift1)/2
        # core_epsmiddle=core_epsmiddle[1:]
        self.energy_corevalue=torch.sum(self.Volume_intergrate(core_eps))
        # core_eps_tobeintergrated=self.raduis[:self.pn_grid_num]**2*core_eps
        # self.energy_corevalue=(2*core_eps_tobeintergrated[1:]-torch.diff(core_eps_tobeintergrated))/2*self.raduisdiff[:self.pn_grid_num-1]*torch.pi*4
        # self.energy_corevalue+=self.raduisinsidecore[0]**2*torch.pi*4*core_eps[0]/2
        # self.energy_corevalue=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*4*torch.pi/3*core_epsmiddle)
        # self.energy_corevalue+=self.raduisinsidecore[0]**3*torch.pi*4/3*core_eps[0]/2
        return self.energy_corevalue
    
    def getmaxrho(self):
        corev=((self.grid_corer**3)*4*torch.pi/3)
        return ((self.bn_n+self.bn_p)/corev)/1.97**3 #单位为 fm-3

            
        
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


    def zhengding(self,a):
        b=a.clone()
        b[-1]=0
        return b**2
    