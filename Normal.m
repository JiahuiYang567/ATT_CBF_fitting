Normal_CBF=load_untouch_nii('wCBF_NII.nii');% read norminalized CBF and ATT
Normal_ATT=load_untouch_nii('wATT_NII.nii');
Normal_CBF=Normal_CBF.img;
Normal_ATT=Normal_ATT.img;
TN_CBF=zeros(95,79,79);
TN_ATT=zeros(95,79,79);
for index_z = 1:79
    TN_CBF(:,:,index_z) = Normal_CBF(:,:,index_z)';
    TN_ATT(:,:,index_z) = Normal_ATT(:,:,index_z)';
end
Whole_CBF=zeros(95*9,79*9);
Whole_ATT=zeros(95*9,79*9);
count=1;
for j=0:7
    for i=0:8
        Whole_CBF((j*95+1):(j*95+95),(i*79+1):(i*79+79))=TN_CBF(:,:,count);
        Whole_ATT((j*95+1):(j*95+95),(i*79+1):(i*79+79))=TN_ATT(:,:,count);
        count=count+1;
    end
end
for last=0:6
    Whole_CBF((8*95+1):(8*95+95),(last*79+1):(last*79+79))=TN_CBF(:,:,count);
    Whole_ATT((8*95+1):(8*95+95),(last*79+1):(last*79+79))=TN_ATT(:,:,count);
    count=count+1;
end
figure;
imagesc(Whole_CBF(:,:),[0 150]);axis off;colorbar;title('Whole brain CBF');
saveas(gcf,'Whole_brain_CBF','jpg');
figure;
imagesc(Whole_ATT(:,:));axis off;colorbar;title('Whole brain ATT');
saveas(gcf,'Whole_brain_ATT','jpg');
Whole_mean_CBF=mean(Whole_CBF(Whole_CBF~=0));
disp(Whole_mean_CBF);
Whole_mean_ATT=mean(Whole_ATT(Whole_ATT~=0));
disp(Whole_mean_ATT);

Baseline_Mask=load_untouch_nii('c1wsaha_pt_S006-0003-00001-000176-01.nii');
% Post_Mask=load_untouch_nii('post_rc1s11136604-0002-00001-000176-01.nii');
Baseline_Mask=Baseline_Mask.img;
Baseline_Mask(Baseline_Mask>216)=255;
Baseline_Mask(Baseline_Mask<=216)=0;
% Post_Mask=Post_Mask.img;
% Post_Mask(Post_Mask>200)=255;
% Post_Mask(Post_Mask<=200)=0;
TN_Mask=zeros(95,79,79);
for index_z = 1:79
    TN_Mask(:,:,index_z)=Baseline_Mask(:,:,index_z)';
end
% Post_Mask=imrotate(Post_Mask,-90);
% P_Mask=zeros(96,96,54);
count=1;
Whole_BMask=zeros(95*9,79*9);
% Whole_PMask=zeros(96*6,96*9);
for j=0:7
    for i=0:8
        Whole_BMask((j*95+1):(j*95+95),(i*79+1):(i*79+79))=TN_Mask(:,:,count);
%         Whole_PMask((j*96+1):(j*96+96),(i*96+1):(i*96+96))=P_Mask(:,:,count);
        count=count+1;
    end
end
for last=0:6
    Whole_BMask((8*95+1):(8*95+95),(last*79+1):(last*79+79))=TN_Mask(:,:,count);
    count=count+1;
end
figure;
imshow(Whole_BMask,[])
whole_mean_CBF_gray=mean(Whole_CBF(Whole_BMask(:,:)==255 & Whole_CBF(:,:)~=0));
whole_mean_ATT_gray=mean(Whole_ATT(Whole_BMask(:,:)==255 & Whole_ATT(:,:)~=0));
% whole_mean_CBF_gray=mean(Whole_CBF(Whole_PMask==255));
disp(whole_mean_CBF_gray);
disp(whole_mean_ATT_gray);