clc; 
clear all
close all

SNR_dB = 5;  
SNR=10.^(SNR_dB/10.);
sigma2=1/SNR;
sample = 1000; 
sparsity=12;

%%% system parameters
N = 512; % number of beams (transmit antennas)
M = 256; % number of pilot overhead


gamma_all=[0:1/3:1];
len = length(gamma_all);

L = 6; % number of all paths


fc = 30e9; % carrier frequency
c = 3e8;
lambda_c = c/fc; % wavelength 
d = lambda_c / 2; % antenna space

% the far-field angle-domain DFT matrix
Uf = (1/sqrt(N))*exp(-1i*pi*[0:N-1]'*[-(N-1)/2:1:(N/2)]*(2/N));

% the near-field polar-domain transform matrix [6]
Rmin=10;
Rmax=80;
eta = 2.5; 
[Un, label, dict_cell, label_cell] = QuaCode(N, d, lambda_c, eta, Rmin, Rmax);

error_omp_dft=zeros(sample,len);
error_omp_qua=zeros(sample,len);
error_homp1=zeros(sample,len);
error_LS=zeros(sample,len);
error_MMSE=zeros(sample,len);
energy=zeros(sample,len);

P=((rand(M,N)>0.5)*2-1)/sqrt(M); % pilot matrix

parfor g=1:len
    g
    gamma=gamma_all(g);
    Lf=gamma*L;
    Ln=L-Lf;
    
    Rh=zeros(N,N);
    for s=1:2000
        [h,hf,hn]=generate_hybrid_field_channel(N, Lf, Ln, d, fc,Rmin, Rmax);
        Rh=Rh+h*h';
    end
    Rh=Rh./(2000);

    for s=1:sample
        s
        [h,hf,hn] = generate_hybrid_field_channel(N, Lf, Ln, d, fc,Rmin, Rmax);
        noise = sqrt(sigma2)*(randn(M,1)+1i*randn(M,1))/sqrt(2);
        y=P*h+noise; 
        %% the far-field OMP based scheme with DFT matrix
        hshat_omp_dft = OMP(y,P*Uf,sparsity*L);
        hhat_omp_dft = Uf*hshat_omp_dft;
        error_omp_dft(s,g)=sum(abs(hhat_omp_dft-h).^2);
        
        %% the near-field OMP based scheme with polar-domain transform matrix
        [hshat_omp_qua,pos_xhat] = OMP(y,P*Un,sparsity*L);
        hhat_omp_qua = Un*hshat_omp_qua;
        error_omp_qua(s,g)=sum(abs(hhat_omp_qua-h).^2);    
        
        %% the proposed hybrid-field OMP based scheme
        hhat_homp=Hybrid_OMP(y,P,Uf,Un,sparsity*Lf,sparsity*Ln);
        error_homp(s,g)=sum(abs(hhat_homp-h).^2);
        
        %% the LS
       hhat_LS=h+sqrt(sigma2)*(randn(N,1)+1i*randn(N,1))/sqrt(2);
       error_LS(s,g)=sum(abs(hhat_LS-h).^2);
       
       %% the MMSE
       hhat_MMSE=Rh*inv(Rh+(sigma2*eye(N)))*hhat_LS; 
       error_MMSE(s,g)=sum(abs(hhat_MMSE-h).^2);
       
       energy(s,g)=sum(abs(h).^2);
    end
end
 
nmse_omp_dft= mean(error_omp_dft)./mean(energy)
nmse_omp_qua = mean(error_omp_qua)./mean(energy)
nmse_homp = mean(error_homp)./mean(energy) 
nmse_LS = mean(error_LS)./mean(energy) 
nmse_MMSE = mean(error_MMSE)./mean(energy) 

nmse_omp_dft=10*log10(nmse_omp_dft)
nmse_omp_qua=10*log10(nmse_omp_qua)
nmse_homp=10*log10(nmse_homp)
nmse_LS=10*log10(nmse_LS)
nmse_MMSE=10*log10(nmse_MMSE)


figure('color',[1,1,1]); 
ha=gca;
plot(gamma_all,nmse_omp_dft,'<-','color',[0.2660 0.9740 0.2880],'linewidth',1.5);
hold on
plot(gamma_all,nmse_omp_qua,'b>-','linewidth',1.5);
hold on
plot(gamma_all,nmse_homp,'rs-','linewidth',1.5);
hold on
plot(gamma_all,nmse_MMSE,'k--','linewidth',1.5);
hold on

grid on
legend('Far-field OMP [3]','Near-field OMP [6]','Proposed hybrid-field OMP','MMSE')
xlabel('\gamma')
ylabel('NMSE (dB)')
hold off

