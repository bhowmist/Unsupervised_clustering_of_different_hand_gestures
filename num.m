% load('OriginalEMG.mat')
 plot(grasp{1,1})
 figure,plot(palm{1,1})
 figure,plot(wave{1,1})
% clc
figure,plot(grasp{1,2})
% figure,plot(grasp{1,3})
% figure,plot(grasp{1,4})
% figure,plot(grasp{1,5})
% clc

X=[];
labels=[];
num_train=20;
for i=1:num_train
 temp=normalizetimeaxis(1000,grasp{i});
 X=[X;temp'];
 labels=[labels;1];
end

for i=1:num_train
 temp=normalizetimeaxis(1000,palm{i});
 X=[X;temp'];
 labels=[labels;2];
end

for i=1:num_train
 temp=normalizetimeaxis(1000,wave{i});
 X=[X;temp'];
 labels=[labels;3];
end

 plot(X(1,:))
 figure,plot(X(21,:))
 figure,plot(X(41,:))

SX=[];                        %spectrogram features
for i=1:size(X,1)
    data=X(i,:);
    [S,f,t,p]=spectrogram(data,128,120,128,1E3);
    psd=10*log10(abs(p));
    surf(psd)
    SX(i,:)=psd(:)';
end
SX(find(SX==-inf))=0;
[coefs,scores,variances,t2]=princomp(SX);

SX(find(SX==-inf))=0;
[coefs,scores,variances,t2]=pca(SX);

figure,
scatter3(scores(1:20,1),scores(1:20,2),scores(1:20,3),'r*'),hold on
scatter3(scores(21:40,1),scores(21:40,2),scores(21:40,3),'k*'),hold on
scatter3(scores(41:60,1),scores(41:60,2),scores(41:60,3),'b*'),hold off

n=10; %no. of features
method= 'diagQuadratic';

graspresult=[];
for i= 1: length(grasp)
    test=normalizetimeaxis(1000,grasp{i});
    [s,f,t,p]=spectrogram(test,128,120,128,1E3);
    psd=10*log10(abs(p));
    
    test=psd(:)';
    test=test-mean(SX); %remove mean
    features=test*coefs;
    
    c=classify(features(1:n),scores(:,1:n),labels,method);
    graspresult(i)=c;
end

palmresult=[];
for i= 1: length(palm)
    test=normalizetimeaxis(1000,palm{i});
    [s,f,t,p]=spectrogram(test,128,120,128,1E3);
    psd=10*log10(abs(p));
    
    test=psd(:)';
    test=test-mean(SX); %remove mean
    features=test*coefs;
    
    c=classify(features(1:n),scores(:,1:n),labels,method);
    palmresult(i)=c;
end


waveresult=[];
for i= 1: length(wave)
    test=normalizetimeaxis(1000,wave{i});
    [s,f,t,p]=spectrogram(test,128,120,128,1E3);
    psd=10*log10(abs(p));
    
    test=psd(:)';
    test=test-mean(SX); %remove mean
    features=test*coefs;
    
    c=classify(features(1:n),scores(:,1:n),labels,method);
    waveresult(i)=c;
end
