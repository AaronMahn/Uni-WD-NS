import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn;
import torch.optim as optim
class rmftorch(nn.Module):
    def __init__(self,modelpar,rho_n,rho_p,bn_n,bn_p,grid_corer,grid_maxr,device):
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
        self.electronmass=torch.tensor(modelpar["electronmass"])
        self.echarge=torch.tensor(modelpar["echarge"])
        #初始密度(迭代参数，一维，球对称，球坐标)
        self.rho_n_rato=nn.Parameter(rho_n).to(torch.device(self.device))
        self.rho_p_rato=nn.Parameter(rho_p).to(torch.device(self.device))
        self.bn_n=bn_n
        self.bn_p=bn_p
        
        # self.rho_p=self.rho_n
        #网格密度（一维）r从0到maxr 间隔为grid_dr 距离都为百mev
        self.grid_corer=grid_corer/1.97     #单位是HMev^-1
        self.pn_grid_num=rho_p.size(0)#0.1个fm距离 换算为了百Mev
        self.grid_incore_dr=self.grid_corer/(self.pn_grid_num-1)
        self.grid_maxr=grid_maxr/1.97       #单位是HMev^-1
        self.ele_grid_num=100    #核外电子取点数量
        self.raduisinsidecore=torch.linspace(start=0.001,end=self.grid_corer,steps=self.pn_grid_num,device=self.device)
        self.raduisoutsidecore=torch.linspace(start=self.grid_corer,end=self.grid_maxr,steps=self.ele_grid_num,device=self.device)
        self.raduis=torch.cat([self.raduisinsidecore,self.raduisoutsidecore])
        self.raduis_rightshift=torch.roll(self.raduis,1)
        self.raduisdiff=(self.raduis-self.raduis_rightshift)
        self.raduisdiff=self.raduisdiff[1:]
        self.raduisdiff_v3=(torch.pow(self.raduis,3)-torch.pow(self.raduis_rightshift,3))
        self.raduisdiff_v3=self.raduisdiff_v3[1:]
        #电磁势（一维，每次迭代需要更新）
        self.A=None
        
    def rho_p(self):
        protonnum=self.zhengding(self.rho_p_rato) #单位为 e*HMev^3
        protonnum_rightshift1=torch.roll(protonnum,1)
        protonchargemiddle=(protonnum+protonnum_rightshift1)/2
        protonchargemiddle=protonchargemiddle[1:]
        totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)
        # print("totalprotonnum:",totalprotonnum)
        print("protonchargemiddle:",protonchargemiddle)
        print("self.raduisdiff_v3[:self.pn_grid_num-1]:",self.raduisdiff_v3[:self.pn_grid_num-1])
        print("self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle:",self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)

        return self.bn_p/totalprotonnum*protonnum
    
    def rho_n(self):
        protonnum=(self.zhengding(self.rho_n_rato)) #单位为 e*HMev^3
        protonnum_rightshift1=torch.roll(protonnum,1)
        protonchargemiddle=(protonnum+protonnum_rightshift1)/2
        protonchargemiddle=protonchargemiddle[1:]
        totalprotonnum=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)
        return self.bn_n/totalprotonnum*protonnum
    def partialsquare_r(self,rho):
        #r=-1 -2 处=r=0处，r=rmax+1 +2 =0 此处理仅适合于强作用场，不适合电磁场，电磁场直接求出无须使用此步骤进行迭代
        
        rho_rightshift1=torch.roll(rho,1)
        rho_rightshift1[0]=rho_rightshift1[1]

        rho_rightshift2=torch.roll(rho,2)
        rho_rightshift2[1]=rho_rightshift2[2]
        rho_rightshift2[0]=rho_rightshift2[1]

        rho_leftshift1=torch.roll(rho,-1)
        rho_leftshift1[-1]=0

        rho_leftshift2=torch.roll(rho,-2)
        rho_leftshift2[-2]=0
        rho_leftshift2[-1]=0
        # print(rho_leftshift2,rho_leftshift1,rho,rho_rightshift2,rho_rightshift1)
        partialsquarer=1/(12*self.grid_incore_dr*self.grid_incore_dr)*(-rho_leftshift2 +16*rho_rightshift1-30*rho+16*rho_leftshift1-rho_rightshift2)
        partialsquarer[0]=0
        partialsquarer[1]=0

        partialr=1/(12*self.grid_incore_dr)*(rho_rightshift2-8*rho_rightshift1+8*rho_leftshift1- rho_leftshift2)
        res=2/self.raduisinsidecore*partialr+partialsquarer
        res[0]=0
        res[1]=0
        res[-1]=0
        res[-2]=0



        # res=res.detach()
        return res
        

    def partialr_square(self,rho):
        #r=-1 -2 处=r=0处，r=rmax+1 +2 =0 此处理仅适合于强作用场，不适合电磁场，电磁场直接求出无须使用此步骤进行迭代

        rho_rightshift1=torch.roll(rho,1)
        rho_rightshift1[0]=rho_rightshift1[1]

        rho_rightshift2=torch.roll(rho,2)
        rho_rightshift2[1]=rho_rightshift2[2]
        rho_rightshift2[0]=rho_rightshift2[1]

        rho_leftshift1=torch.roll(rho,-1)
        rho_leftshift1[-1]=0

        rho_leftshift2=torch.roll(rho,-2)
        rho_leftshift2[-2]=0
        rho_leftshift2[-1]=0

        # partialr=1/(12*self.grid_incore_dr)*(rho_leftshift2-8*rho_leftshift1+8*rho_rightshift1-rho_rightshift2)
        partialr=1/(12*self.grid_incore_dr)*(rho_rightshift2-8*rho_rightshift1+8*rho_leftshift1- rho_leftshift2)
        # partialr=partialr.detach()
        return partialr*partialr

    def findsigma(self,startpoint=0.0,step=0.1,accuracy=4):
        #在迭代的时候可能会自动梯度下降到小于0的状态，所以要进行检查一下
        rho_p=self.zhengding(self.rho_p())
        rho_n=self.zhengding(self.rho_n())

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
            newpartialsigma=self.partialsquare_r(sigma)
            partialsigma=partialsigma+(newpartialsigma-partialsigma)*step
            cout=cout+1

        # print('sigma:cout=',cout)
        
        return sigma,partialsigma
    def findomega(self,startpoint=0.0,step=0.1,accuracy=4):
        #在迭代的时候可能会自动梯度下降到小于0的状态，所以要进行检查一下
        rho_p=self.zhengding(self.rho_p())
        rho_n=self.zhengding(self.rho_n())
        #设置精度：
        pointnum=rho_p.size()

        deltaomega=10000
        omega=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        partialomega=(torch.rand(pointnum)*startpoint).to(torch.device(self.device))
        cout=0
        while torch.sum(((abs(deltaomega))>abs(omega)*torch.pow(torch.tensor(0.1),accuracy)))>0:

            omeganew=(self.g_omega*(rho_p+rho_n)+partialomega)/torch.pow(self.m_omega,2)
            deltaomega=omeganew-omega
            omega=omega+deltaomega*step
            newpartialomega=self.partialsquare_r(omega)
            partialomega=partialomega+(newpartialomega-partialomega)*step
            cout=cout+1
            # print(omeganew)
        # print('omega:cout=',cout)
        return omega,partialomega
    def findrho(self,startpoint=0.0,step=0.01,accuracy=4):
        #在迭代的时候可能会自动梯度下降到小于0的状态，所以要进行检查一下
        rho_p=self.zhengding(self.rho_p())
        rho_n=self.zhengding(self.rho_n())
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
            newpartialrho=self.partialsquare_r(rho)
            partialrho=partialrho+(newpartialrho-partialrho)*step
            cout=cout+1
            
        # print('rho:cout=',cout)
        return rho,partialrho

    def update_electricfield(self):
        #

        # print('raduisinsidecore:',self.raduisinsidecore)
        # print('raduisdiff_v3:',torch.sum(raduisdiff_v3))
        protoncharge=(self.zhengding(self.rho_p())) #单位为 e*HMev^3
        protoncharge_rightshift1=torch.roll(protoncharge,1)
        protonchargemiddle=(protoncharge+protoncharge_rightshift1)/2
        protonchargemiddle=protonchargemiddle[1:]

        totalprotoncharge=torch.sum(self.raduisdiff_v3[:self.pn_grid_num-1]*torch.pi*4/3*protonchargemiddle)
        core_out_v=(self.grid_maxr**3-self.grid_corer**3)*4/3*torch.pi#核外体积
        # print('core_out_v:',core_out_v)
        electrondensity=totalprotoncharge/core_out_v#电子密度
        # totalgridnum=self.ele_grid_num+self.rho_p.size(0)
        totaldensitydis=torch.cat([protonchargemiddle,-torch.ones(self.ele_grid_num,device=self.device)*electrondensity],dim=0)

        elecharge=totaldensitydis*self.raduisdiff_v3*4*torch.pi/3#单位是e
        eleforce=torch.zeros(elecharge.size(),device=self.device)
        
        raduis=self.raduis[1:]
        # print("raduis:",raduis)
        for i in range(elecharge.size(0)):
            eleforce[i]=torch.sum(elecharge[:i+1])/(4*torch.pi*raduis[i]*raduis[i]) #单位是e*HMev^2
        # print("protoncharge:",protoncharge)
        # print("electrondensity:",electrondensity)
        # print('totalelectroncharge:',torch.sum(elecharge[self.pn_grid_num-1:]))
        # print('totalprotoncharge:',totalprotoncharge,torch.sum(elecharge[:self.pn_grid_num-1]))
        # print('totalcharge:',torch.sum(elecharge))
        # print("elecharge:",elecharge)
        # print("totaldensitydis:",totaldensitydis)
        # print("raduisdiff_v3*4*torch.pi/3:",raduisdiff_v3*4*torch.pi/3)
        # print("eleforce:",eleforce)
        elepdiff=(eleforce*self.raduisdiff)#单位是e*HMeV
        # corep_surface=elep[self.pn_grid_num-1:]#gridnum-1是因为段数比点数少1
        elep=torch.zeros(protoncharge.size(0),device=self.device)
        for i in range(protoncharge.size(0)):
            elep[i]=torch.sum(elepdiff[i:])

        # elep=torch.zeros(totalgridnum,device=self.device)
        # for i in range(totalgridnum):
        #     elep[i]=torch.sum(elepdiff[i:])
        self.A=elep*self.echarge#单位是e*HMeV
        
    def eps(self,step=0.1):
        rho_p=self.zhengding(self.rho_p())
        rho_n=self.zhengding(self.rho_n())
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
        partialsigma=self.partialr_square(sigma)
        partialomega=self.partialr_square(omega)
        partialrho=self.partialr_square(rho)
        partialA=self.partialr_square(self.A)
        # eps=torch.pow(m_n,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*(a1+a2)+self.g_omega*omega*(rho_n+rho_p)+self.g_rho*rho*(rho_p-rho_n)+0.5*torch.pow(self.m_sigma,2)*torch.pow(sigma,2)+1/3*self.g2*torch.pow(sigma,3)+0.25*self.g3*torch.pow(sigma,4)-0.5*torch.pow(self.m_omega,2)*torch.pow(omega,2)-0.5*torch.pow(self.m_rho*rho,2)+0.5*(partialsigma+partialomega+partialrho+partialA)
        eps=torch.pow(m_n,4)/(8*torch.pow(torch.tensor(torch.pi, requires_grad=True),2))*(a1+a2)+0.5*torch.pow(self.m_omega,2)*torch.pow(omega,2)+0.5*torch.pow(self.m_rho*rho,2)+0.5*torch.pow(self.m_sigma,2)*torch.pow(sigma,2)+1/3*self.g2*torch.pow(sigma,3)+0.25*self.g3*torch.pow(sigma,4)+0.5*(partialsigma+partialomega+partialrho+partialA)

        return eps
    def epsingral(self,step=0.1):
        return torch.sum(self.raduisdiff_v3[:self.pn_grid_num]*4*torch.pi/3*self.eps(step=step))
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
        rho_p=self.zhengding(self.rho_p)
        rho_n=self.zhengding(self.rho_n)
        
        cp_p=self.g_omega*omega+self.g_rho*rho+self.echarge*self.A+torch.pow(torch.pow(3*torch.pi*torch.pi*rho_p,2/3)+torch.pow(self.nucleonmass+self.g_sigma*sigma,2),0.5)
        cp_n=self.g_omega*omega-self.g_rho*rho+torch.pow(torch.pow(3*torch.pi*torch.pi*rho_n,2/3)+torch.pow(self.nucleonmass+self.g_sigma*sigma,2),0.5)
        return cp_n*rho_n+cp_p*rho_p
    def constancycondition(self,cp):
        return torch.sum(torch.abs(self.partial_periodicity(cp,self.partialstep)))
    def zhengding(self,a):
        b=a.clone()
        for i in range(a.size(0)):
                    if a[i]<0:
                        b[i]=0.0001
                    else:
                        b[i]=a[i]
        return b
    