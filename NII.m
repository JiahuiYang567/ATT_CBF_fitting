% Try one slice from UMN data
% Similar idea from T1_fitting_MDASL_simulation
% While T1 mapping only uses control images, include tagged images to
% estimate T1 and CBF simultaneously, both control/tag image will impose
% constraints on T1 fitting, should generate better results.
% Now try to estimate ATT simultaneously! Reference to: Multi-delay multi-parametric arterial spin-labeled perfusion MRI in acute ischemic stroke ? Comparison with dynamic susceptibility contrast enhanced perfusion imaging

% 11072016, try this for Multidelay data with 4-fold under-sampling using
% ESPIRIT recon

function NII
%% Basic parameters
clear;
close all;
clc;
tic;

Im=zeros(96,96,54,16);
%post
% Im1=load_untouch_nii('f11136604-0014-00016-000811-01.nii');
% Im2=load_untouch_nii('f11136604-0014-00015-000757-01.nii');
% Im3=load_untouch_nii('f11136604-0014-00014-000703-01.nii');
% Im4=load_untouch_nii('f11136604-0014-00013-000649-01.nii');
% Im5=load_untouch_nii('f11136604-0014-00012-000595-01.nii');
% Im6=load_untouch_nii('f11136604-0014-00011-000541-01.nii');
% Im7=load_untouch_nii('f11136604-0014-00010-000487-01.nii');
% Im8=load_untouch_nii('f11136604-0014-00009-000433-01.nii');
% Im9=load_untouch_nii('f11136604-0014-00008-000379-01.nii');
% Im10=load_untouch_nii('f11136604-0014-00007-000325-01.nii');
% Im11=load_untouch_nii('f11136604-0014-00006-000271-01.nii');
% Im12=load_untouch_nii('f11136604-0014-00005-000217-01.nii');
% Im13=load_untouch_nii('f11136604-0014-00004-000163-01.nii');
% Im14=load_untouch_nii('f11136604-0014-00003-000109-01.nii');
% Im15=load_untouch_nii('f11136604-0014-00002-000055-01.nii');
% Im16=load_untouch_nii('f11136604-0014-00001-000001-01.nii');
%baseline
Im1=load_untouch_nii('faha_pt_S006-0025-00016-000811-01.nii');
Im2=load_untouch_nii('faha_pt_S006-0025-00015-000757-01.nii');
Im3=load_untouch_nii('faha_pt_S006-0025-00014-000703-01.nii');
Im4=load_untouch_nii('faha_pt_S006-0025-00013-000649-01.nii');
Im5=load_untouch_nii('faha_pt_S006-0025-00012-000595-01.nii');
Im6=load_untouch_nii('faha_pt_S006-0025-00011-000541-01.nii');
Im7=load_untouch_nii('faha_pt_S006-0025-00010-000487-01.nii');
Im8=load_untouch_nii('faha_pt_S006-0025-00009-000433-01.nii');
Im9=load_untouch_nii('faha_pt_S006-0025-00008-000379-01.nii');
Im10=load_untouch_nii('faha_pt_S006-0025-00007-000325-01.nii');
Im11=load_untouch_nii('faha_pt_S006-0025-00006-000271-01.nii');
Im12=load_untouch_nii('faha_pt_S006-0025-00005-000217-01.nii');
Im13=load_untouch_nii('faha_pt_S006-0025-00004-000163-01.nii');
Im14=load_untouch_nii('faha_pt_S006-0025-00003-000109-01.nii');
Im15=load_untouch_nii('faha_pt_S006-0025-00002-000055-01.nii');
Im16=load_untouch_nii('faha_pt_S006-0025-00001-000001-01.nii');
%
% MASK = load_nii([targetname,'_all_highres_brain_mask.nii.gz']); % brain mask
Im(:,:,:,1) = Im16.img;
Im(:,:,:,2) = Im15.img;
Im(:,:,:,3) = Im14.img;
Im(:,:,:,4) = Im13.img;
Im(:,:,:,5) = Im12.img;
Im(:,:,:,6) = Im11.img;
Im(:,:,:,7) = Im10.img;
Im(:,:,:,8) = Im9.img;
Im(:,:,:,9) = Im8.img;
Im(:,:,:,10) = Im7.img;
Im(:,:,:,11) = Im6.img;
Im(:,:,:,12) = Im5.img;
Im(:,:,:,13) = Im4.img;
Im(:,:,:,14) = Im3.img;
Im(:,:,:,15) = Im2.img;
Im(:,:,:,16) = Im1.img;

Im_M0 = Im16.img;

% Average 4repeated images first
% Im_tmp(:,:,:,[1 2]) = Im_M0;
Im_tmp(:,:,:,[1 2]) = Im(:,:,:,[1 2]);

% for i=1:(size(Im,4)-2)/4
%     Im_tmp(:,:,:,2+i) = mean(Im(:,:,:,3+4*(i-1):2:2+4*i),4);
% end
Im_tmp(:,:,:,3) = Im(:,:,:,3); % PLD 300 label
Im_tmp(:,:,:,4) = Im(:,:,:,4); % PLD 300 control
Im_tmp(:,:,:,5) = Im(:,:,:,5);% PLD 800 label
Im_tmp(:,:,:,6) = Im(:,:,:,6);
Im_tmp(:,:,:,7) = Im(:,:,:,7);% PLD 1300 label
Im_tmp(:,:,:,8) = Im(:,:,:,8);
Im_tmp(:,:,:,9) = (Im(:,:,:,9)+Im(:,:,:,11))/2;% PLD 1800 label
Im_tmp(:,:,:,10) = (Im(:,:,:,10)+Im(:,:,:,12))/2;
Im_tmp(:,:,:,11) = (Im(:,:,:,13)+Im(:,:,:,15))/2;% PLD 2500 label
Im_tmp(:,:,:,12) = (Im(:,:,:,14)+Im(:,:,:,16))/2;
% Im_tmp(:,:,:,15) = Im(:,:,:,15);% PLD 2500 label
% Im_tmp(:,:,:,16) = Im(:,:,:,16);

Im=Im_tmp;
% Im(:,64,:,:)=0;% add or not?

MASK = zeros(size(Im,1),size(Im,2),size(Im,3));
MASK(Im(:,:,:,1)>2*mean(mean(mean(Im(:,:,:,1)))))=1;
% figure,imagesc(im2dis(Im(:,:,:,1).*MASK,6,8));
% figure,imagesc(im2dis(MASK,6,8));


% relative image from control series
[nx,ny,nz,nt] = size(Im);
T1e = zeros(nx,ny,nz);
T1_default = 1000;
T1_lb  = 500;
T1_ub  = 1500;
rIm = zeros(nx,ny,nz,nt/2-1);
dIm = zeros(nx,ny,nz,nt/2-1);
for index_t = 4:2:nt % Control image starts from the fourth one
    rIm(:,:,:,(index_t)/2-1) = double(Im(:,:,:,index_t)./Im(:,:,:,1).*MASK); % Relative signal for calculation
end

% Assume negative background signal is most likely noise, set them to 0
rIm(rIm<0)=0;

% Diff images, perfusion!
for index_t = 1:(nt/2-1)
    dIm(:,:,:,index_t) = double((Im(:,:,:,2+2*index_t) - Im(:,:,:,1+2*index_t))./Im(:,:,:,1).*MASK);
end

% Assume negative dif signal is most likely noise, set them to 0
dIm(dIm<0)=0;

%% BS parameters
%  BS delay was calculated as 0.8063*(PLD - 50 or 75) - 63.05
%  Duration of BS pulse: 0.5 dummy <-- 10.24 pulse <-- 0.5 dummy <-- 5.5
%  spoiler?? So adjust tsat, tinv, PLD accordingly. Also should include fat
%  sat duration 12.2+1.19 = 13.39
global M0;
M0 = 1;
BS_extra = 0.5+10.24;
BSpulse_extra = 16.74;
fatsat_extra  = 13.39;
LD = 1500; %labeling duration
PLD = [300 800 1300 1800 2300]; % UI, also serve as the second inversion time
NP_shift = [50 50 50 50 75]; % hard-coded, effective since 061916, used for UMN HCP
BS_delay = floor(0.8063.*(PLD - NP_shift) - 63.05);
tinv = PLD - BS_delay + fatsat_extra+ BSpulse_extra - BS_extra;
tsat = PLD + LD + fatsat_extra + BSpulse_extra*2; % check
PLD  = PLD + fatsat_extra + BSpulse_extra*2 - BS_extra; %check

%% CBF parameters
global lamda R1a alpha tau
lamda = 0.9;   % blood/tissue water partition coefficient, g/ml
R1a   = 0.606; % longitudinal relaxation rate of blood
alpha = 0.767; % labeling efficiency
tau   = 1.500; % labeling duration
CBFe = zeros(nx,ny,nz);
CBF_default = 80 ; % is this right?
CBF_ub  = 200;
CBF_lb  = 0.1;
ATTe = zeros(nx,ny,nz);
ATT_default = 0.7; % defalut 0.7s according to BASIL implementation
ATT_lb  = 0.5;
ATT_ub  = 2.5;

%% CBF_ATT from multidelay multi-parametric paper
%  Get ATT from a look up table: ATT_WD_pool = [ATT;WD], ATT = 500:1:2500
S=load('ATT-WD.mat');
ATT_pool=S.ATT_pool;
WD_pool=S.WD_pool;
ATTr = zeros(nx,ny,nz); % XXXr stands for reference methods
dIm_temp = dIm;
dIm_temp(dIm<0)=0; % get rid of negative points for ATT_WD estimation

for index_x = 1:nx
    for index_y = 1:ny
        for index_z = 1:nz
            ATT_temp=0;% if MASK==0 No ATT_temp?
            if MASK(index_x,index_y,index_z) ~= 0
                
                % Calculate within ROI
                WD_temp = round(sum(squeeze(dIm_temp(index_x,index_y,index_z,:)).*PLD')./.../
                    sum(squeeze(dIm_temp(index_x,index_y,index_z,:))));% WD function
                
                if isnan(WD_temp) % if there is a NaN in wd_temp, ATT_temp=min(ATT_pool)
                    ATT_temp = min(ATT_pool);
                end
                if WD_temp<=min(WD_pool)
                    ATT_temp = min(ATT_pool);
                elseif WD_temp>=max(WD_pool)
                        ATT_temp = max(ATT_pool);
                    else
                        if (~isempty(find(WD_pool==round(WD_temp),1)))% if WD_temp in WD_pool,
                            pos_temp = find(WD_pool==round(WD_temp));% Return the coordinate of the same one in the ATT_pool.
                            ATT_temp = ATT_pool(pos_temp(1));% use WD find ATT
                        end
                end
            end
            ATTr(index_x,index_y,index_z) = ATT_temp;
            % ATT_temp=0;
        end
    end
end

CBFw = zeros(nx,ny,nz,(nt/2-1));
ASL = squeeze(dIm(:,:,:,:));
w = PLD/1000;
for index_x = 1:nx
    for index_y = 1:ny
        for index_z = 1:nz
            if MASK(index_x,index_y,index_z) ~= 0
                delta = ATTr(index_x,index_y,index_z)/1000;
                CBFw(index_x,index_y,index_z,:) = 60*100*lamda.*squeeze(ASL(index_x,index_y,index_z,:))'.*R1a./(2*alpha.*M0.*(exp((min(delta-w,0)-delta).*R1a)-exp(-(tau+w)*R1a)));%CBF at each post labeling delay,use ASL to represent dM
            end
        end
    end
end

CBFr = mean(CBFw,4);
% figure,imagesc(CBFr(:,:,20));colorbar;

% Generate default value for fitting approach
CBF_default_map = CBFr;
ATT_default_map = ATTr;

% Roatate image for display
Im_CBF=Im1;
Im_ATT=Im1;
a=zeros(nx,ny,nz);
b=zeros(nx,ny,nz);
for index_z = 1:nz
    a(:,:,index_z) = CBFr(:,:,index_z);
    b(:,:,index_z) = ATTr(:,:,index_z);
end
for indz=1:nz
    for indy=1:ny
        for indx=1:nx
            if a(indx,indy,indz)>150
                a(indx,indy,indz)=150;
            end
            if a(indx,indy,indz)<0
                a(indx,indy,indz)=0;
            end
        end
    end
end
Im_CBF.img=uint16(a);
Im_ATT.img=uint16(b);
save_untouch_nii(Im_CBF,'CBF_NII.nii');
save_untouch_nii(Im_ATT,'ATT_NII.nii');
% Finish generating CBF and ATT nii, then use spm to generate norminalized
% CBF and ATT

Normal_CBF=load_untouch_nii('wCBF_NII.nii');% read norminalized CBF and ATT
Normal_ATT=load_untouch_nii('wATT_NII.nii');
Normal_CBF=Normal_CBF.img;
Normal_ATT=Normal_ATT.img;
for index_z = 1:79
    Normal_CBF(:,:,index_z) = Normal_CBF(:,:,index_z)';
    Normal_ATT(:,:,index_z) = Normal_ATT(:,:,index_z)';
end
Whole_CBF=zeros(95*9,79*9);
Whole_ATT=zeros(95*9,79*9);
count=1;
for j=0:7
    for i=0:8
        Whole_CBF((j*95+1):(j*95+95),(i*79+1):(i*79+79))=Normal_CBF(:,:,count);
        Whole_ATT((j*95+1):(j*95+95),(i*79+1):(i*79+96))=Normal_ATT(:,:,count);
        count=count+1;
    end
end
for last=0:6
    Whole_CBF((8*95+1):(8*95+95),(last*79+1):(last*79+79))=Normal_CBF(:,:,count);
    Whole_ATT((8*95+1):(8*95+95),(last*79+1):(last*79+79))=Normal_ATT(:,:,count);
    count=count+1;
end
figure;
imagesc(Whole_CBF(:,:),[0 150]);axis off;colorbar;title('Whole brain CBF');
saveas(gcf,'Whole_brain_CBF','jpg');
figure;
imagesc(Whole_ATT(:,:));axis off;colorbar;title('Whole brain ATT');
saveas(gcf,'Whole_brain _ATT','jpg');
Whole_mean_CBF=mean(Whole_CBF(Whole_CBF~=0));
disp(Whole_mean_CBF);
Whole_mean_ATT=mean(Whole_ATT(Whole_ATT~=0));
disp(Whole_mean_ATT);

Baseline_Mask=load_untouch_nii('c1ws180503153326DST131221107524366096-0002-00001-000176-01.nii');
% Post_Mask=load_untouch_nii('post_rc1s11136604-0002-00001-000176-01.nii');
Baseline_Mask=Baseline_Mask.img;
Baseline_Mask(Baseline_Mask>216)=255;
Baseline_Mask(Baseline_Mask<=216)=0;
% Post_Mask=Post_Mask.img;
% Post_Mask(Post_Mask>200)=255;
% Post_Mask(Post_Mask<=200)=0;
for index_z = 1:79
    Baseline_Mask(:,:,index_z)=Baseline_Mask(:,:,index_z)';
end
% Post_Mask=imrotate(Post_Mask,-90);
% P_Mask=zeros(96,96,54);
count=1;
Whole_BMask=zeros(95*9,79*9);
% Whole_PMask=zeros(96*6,96*9);
for j=0:7
    for i=0:8
        Whole_BMask((j*95+1):(j*95+95),(i*79+1):(i*79+79))=Baseline_Mask(:,:,count);
%         Whole_PMask((j*96+1):(j*96+96),(i*96+1):(i*96+96))=P_Mask(:,:,count);
        count=count+1;
    end
end
figure;
imshow(Whole_BMask,[])
whole_mean_CBF_gray=mean(Whole_CBF(Whole_BMask(:,:)==255 & Whole_CBF(:,:)~=0));
whole_mean_ATT_gray=mean(Whole_ATT(Whole_BMask(:,:)==255 & Whole_ATT(:,:)~=0));
% whole_mean_CBF_gray=mean(Whole_CBF(Whole_PMask==255));
disp(whole_mean_CBF_gray);
disp(whole_mean_ATT_gray);
% whole_mean_CBF_gray=mean(Whole_ATT(Post_Mask>=0.8));
% save('post_CBF_ATT.mat','Whole_CBF','Whole_ATT');
% Axial display
% figure,imagesc(im2dis(CBFr,7,7),[0 150]);axis off;colorbar;title(['Whole brain CBF_axial: ',targetname]);

% Sagittal display
% CBFr_sag = permute(CBFr,[3 2 1]);
% CBFr_sag(:,:,97:100) = zeros(49,96,4);
% figure,imagesc(im2dis(imrotate(CBFr_sag(:,:,30:10:70),-90),1,5),[0 150]);axis off;colorbar;title(['Whole brain CBF sag: ',targetname]);

% Coronal display
% CBFr_cor = permute(CBFr,[1 3 2]);
% CBFr_cor(:,:,97:100) = zeros(96,49,4);
% figure,imagesc(im2dis(imrotate(CBFr_cor(:,:,30:10:70),180),1,5),[0 150]);axis off;colorbar;title(['Whole brain CBF cor: ',targetname]);
% CBFr_corsag2dis = imrotate(CBFr_sag(:,:,30:10:70),-90);
% CBFr_corsag2dis(:,:,6:10) = imrotate(CBFr_cor(:,:,30:10:70),180);
% figure,imagesc(im2dis(imrotate(CBFr_corsag2dis(:,:,:),0),1,10),[0 150]);axis off;colorbar;%title(['Whole brain CBF cor: ',targetname]);


% figure,imagesc(im2dis(ATTr/1000,6,8));axis off;colorbar;title(['Whole brain ATT: ',targetname]);

%% T1_CBF fitting here!!
  BS_timing = [tsat;tinv;PLD];
for index_x = 1:nx
    for index_y = 1:ny
        for index_z = 1:nz
            if (MASK(index_x,index_y,index_z) ~= 0 && isempty(find(isnan(squeeze((rIm(index_x,index_y,index_z,:)))),1)).../
                    && isempty(find(isnan(squeeze((dIm(index_x,index_y,index_z,:)))),1)))%mask is 1 and rIm & dIm have multipule delay 
                
                % Get initial value from above calcuated results
                CBF_default = CBF_default_map(index_x,index_y,index_z);
                ATT_default = ATT_default_map(index_x,index_y,index_z)/1000; % add /1000!!!
                
                T1_CBF_temp = lsqcurvefit(@T1_CBF_ATT_fitting_function_main,.../
                    [T1_default,CBF_default,ATT_default],BS_timing,.../
                    [squeeze((rIm(index_x,index_y,index_z,:))),squeeze((dIm(index_x,index_y,index_z,:)))],.../
                    [T1_lb, CBF_lb, ATT_lb],[T1_ub, CBF_ub, ATT_ub]);
                T1e(index_x,index_y,index_z) = T1_CBF_temp(1);
                CBFe(index_x,index_y,index_z) = T1_CBF_temp(2);
                ATTe(index_x,index_y,index_z) = T1_CBF_temp(3);
            end
        end
    end
end

% rotate the image for display
for index_z = 1:nz
    T1e(:,:,index_z) = T1e(:,:,index_z)';
    CBFe(:,:,index_z) = CBFe(:,:,index_z)';
    ATTe(:,:,index_z) = ATTe(:,:,index_z)';
end
Whole_CBFe=zeros(96*6,96*9);
Whole_ATTe=zeros(96*6,96*9);
Whole_T1e=zeros(96*6,96*9);
count=1;
for j=0:5
    for i=0:8
        Whole_CBFe((j*96+1):(j*96+96),(i*96+1):(i*96+96))=CBFe(:,:,count);
        Whole_ATTe((j*96+1):(j*96+96),(i*96+1):(i*96+96))=ATTe(:,:,count);
        Whole_T1e((j*96+1):(j*96+96),(i*96+1):(i*96+96))=T1e(:,:,count);
        count=count+1;
    end
end
figure,imagesc(Whole_T1e(:,:),[700,1000]);axis off;colorbar;title('Whole brain fitted T1 map: ');colormap jet
figure,imagesc(Whole_CBFe(:,:),[0 150]);axis off;colorbar;title('Whole brain fitted CBF: ');
figure,imagesc(Whole_ATTe(:,:));axis off;colorbar;title('Whole brain fitted ATT: ');

Whole_mean_CBFe=mean(Whole_CBFe(Whole_CBFe~=0));
disp(Whole_mean_CBFe);
Whole_mean_ATTe=mean(Whole_ATTe(Whole_ATTe~=0));
disp(Whole_mean_ATTe);
whole_mean_CBFe_gray=mean(Whole_CBFe(Whole_BMask(:,:)==255 & Whole_CBFe(:,:)~=0));
disp(whole_mean_CBFe_gray);

toc;
toc; % Done! total time for subject 4 is 4100 s


end % end of main function
function ydata = T1_CBF_ATT_fitting_function_main(xdata,BS_timing)
global M0 lamda R1a alpha tau
tsat = BS_timing(1,:);%time for application saturation pulse.
tinv = BS_timing(2,:);%time for application inverse pulse.
PLD  = BS_timing(3,:);
w = PLD/1000; % For CBF fitting
% T1 fitting -- ydata(1,:) refers to background suppressed signal
T1 = xdata(1);
ydata(1,:) = M0.*(1-exp(-tsat./T1)-2.*exp(-tinv./T1)+2.*exp(-PLD./T1));%do not know what is this, MT1?

% CBF fitting -- yadata(2,:) refers to the signal difference (Control - Label)
% CBF calculation based on: CBF = 60*100*lamda.*ASL.*R1a./(2*alpha.*M0.*(exp(-PLD.*R1a)-exp(-(tau+PLD)*R1a)));
CBF = xdata(2);
ATT = xdata(3);
ydata(2,:) = max(0,CBF./(60*100*lamda.*R1a).*(2*alpha.*M0.*(exp((min(ATT-w,0)-ATT).*R1a)-exp(-(tau+w)*R1a))));%ATT=dM
ydata = ydata';
end