function [H,iter]=l2l2_resection(u,U,epsperc,maxerror,maxiter);
% Computes optimal homography or projection matrix of size (m+1)x(n+1), where m<=n
% from source points U in P^n to target points in P^m
% Inputs:
%   u - non-homogeneous target coordinates, for example, 2xp matrix for p
%   image points
%   U - non-homogeneous source coordinates, for example, 3xp matrix for p
%   world points
% Outputs:
%   H - (m+1)x(n+1) matrix corresponding to a homography (m=n) or
%   projection (m<n)


fprintf('\n\n******** Starting (L2,L2) Resectioning Algorithm ********\n\n');

if nargin<5 | isempty(maxiter),
    maxiter=50;
end

if nargin<4 | isempty(maxerror),
    maxerror=15;
end
if nargin<3 || isempty(epsperc),
    epsperc=0.05;
else;
	epsperc = 1-epsperc;
end
%fprintf('EPS : %f\n',epsperc);

sourceD=size(U,1);
imageD=size(u,1);

nbrpoints=size(u,2);
%epsdiff=1e-7;
epsdiff=0.0;





%translate image centroid to origine (and rescale coordinates)
mm=mean(u')';
ut=u-mm*ones(1,nbrpoints);
imscale=std(ut(:));
ut=ut/imscale;


K=diag([ones(1,imageD)/imscale,1]);K(1:imageD,end)=-mm/imscale;

%translate source points such that U_1=[0,...,0,1]^T (and rescale coordinates)
mm=U(:,1);
Ut=U-mm*ones(1,nbrpoints);
tmp=Ut(:,2:end);
ss=std(tmp(:));
Ut=Ut/ss;

T=diag([ones(1,sourceD)/ss,1]);T(1:sourceD,end)=-mm/ss;


%upper & lower bound...
thr = maxerror/imscale; %maximum number of pixels for a single term
[rect_yL,rect_yU]=l2_reconstruct_bound(ut,Ut,thr);
xL=0;xU=2*thr;



hh=l2_reconstruct_loop(ut,Ut,rect_yL,rect_yU,xL,xU);
%make better estimate of xU based on hh.H
tmp=hh.H*[Ut;ones(1,nbrpoints)];
tmp=tmp(1:2,:)./(ones(2,1)*tmp(3,:))-ut;
xU=max(abs(tmp(:))/min(rect_yL))*2;



vopt=sum(hh.res);

H=hh.H;

rect_LB=hh.lowerbound;
rect={hh};


iter=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iter <= maxiter, %Branch and Bound-loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [vk,vkindex]=min(rect_LB);
   
   vdiff=(vopt-vk);
   perc=(vopt-vk)/vopt;
   
   disp(['Iter: ',num2str(iter),' Residual: ',num2str(vopt*imscale),' Approximation gap: ',num2str(perc*100),'% Regions: ',num2str(length(rect))]);

   if vdiff<epsdiff || perc<epsperc,
       %'voila'
       break;
   end
   
   %branch on vkindex
   h=rect{vkindex};
%   index_z=3+nbrimages+[1:3:3*(nbrimages-1)]; %location of z_2,z_3,...,z_nbrimages
%   index_z=3+nbrimages+[1:3:3*(min(nbrimages,4)-1)]; %location of z_2,z_3,...,z_nbrimages

   %denominator to branch on
%   [slask,pp]=max(h.res(2:min(nbrimages,4))'-h.y(index_z));
%   [slask,pp]=max(h.res(2:nbrimages)'-h.y(index_z));
   [slask,pp]=max(rect_yU(1:sourceD,vkindex)-rect_yL(1:sourceD,vkindex)); %largest interval
   
   tmpyL=rect_yL(pp,vkindex);
   tmpyU=rect_yU(pp,vkindex);
   
   %branching strategy....
   
   bestsol=h.lambda(pp);
   alfa=0.2; %minimum shortage of interval relative original
   if (bestsol-tmpyL)/(tmpyU-tmpyL)<alfa,
       newborder=tmpyL+(tmpyU-tmpyL)*alfa;
   elseif (tmpyU-bestsol)/(tmpyU-tmpyL)<alfa;
       newborder=tmpyU-(tmpyU-tmpyL)*alfa;
   else
       newborder=bestsol;
   end
   
   
%   bisect=(tmpyU+tmpyL)/2;
%   bestsol=h.lambda(pp);
%   alfa=0.8;
%   newborder=bestsol*alfa+bisect*(1-alfa);
   
%   newborder=(tmpyU+tmpyL)/2; %bisection
   
%       newborder=h.lambda(pp); %best solution


   curr_yL1=rect_yL(:,vkindex);
   curr_yU1=rect_yU(:,vkindex);
   
   curr_yL2=curr_yL1;
   curr_yU2=curr_yU1;
   
   curr_yU1(pp)=newborder;
   curr_yL2(pp)=newborder;
   
   rect_yL=[rect_yL(:,1:vkindex-1),curr_yL1,curr_yL2,rect_yL(:,vkindex+1:end)];
   rect_yU=[rect_yU(:,1:vkindex-1),curr_yU1,curr_yU2,rect_yU(:,vkindex+1:end)];
   
   h1=l2_reconstruct_loop(ut,Ut,curr_yL1,curr_yU1,xL,xU);
   h2=l2_reconstruct_loop(ut,Ut,curr_yL2,curr_yU2,xL,xU);
   
   vopt1=sum(h1.res);
   vopt2=sum(h2.res);
   
   rect={rect{1:vkindex-1},h1,h2,rect{vkindex+1:end}};
   rect_LB=[rect_LB(1:vkindex-1),h1.lowerbound,h2.lowerbound,rect_LB(vkindex+1:end)];
   
   if vopt1<vopt,
       vopt=vopt1;
       H=h1.H;
   end
   if vopt2<vopt,
       vopt=vopt2;
       H=h2.H;
   end
   
   %screen and remove useless regions
   removeindex=[];
   for ii=1:length(rect);
       if rect{ii}.lowerbound>vopt,
           %remove!
           removeindex(end+1)=ii;
       end
   end
   rect(removeindex)=[];
   rect_yL(:,removeindex)=[];
   rect_yU(:,removeindex)=[];
   rect_LB(removeindex)=[];
       
   iter=iter+1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end, %Branch and Bound-loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


H=inv(K)*H*T;
H=H/norm(H);

fprintf('******** Ending (L2,L2) Resectioning Algorithm ********\n\n');
return



function [rect_yL,rect_yU]=l2_reconstruct_bound(ut,Ut,thr,rect_yL,rect_yU,updateindex);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find bounds on depths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxU=5; %maximum depth upper bound
minL=0.2; %minimum depth lower bound
sourceD=size(Ut,1);
imageD=size(ut,1);
nbrpoints=size(ut,2);

if nargin<4,
    rect_yL=minL*ones(nbrpoints-1,1);
    rect_yU=maxU*ones(nbrpoints-1,1);
    updateindex=1:nbrpoints-1;
end


%variable order:
% last row [h_n1,h_n2,...] where it is assumed h_nn=1
% first row, second row, etc
% upper bounds a_2,...,a_n on numerators

Hvars= (sourceD+1)*(imageD+1)-1;
vars = Hvars + nbrpoints-1;


%sedumi matrices
At_l=sparse(zeros(0,vars)); %linear inequalities
c_l=sparse(zeros(0,1));
At=sparse(zeros(0,vars)); %cone constraints
c=sparse(zeros(0,1));
clear K;
K.l=0;
K.q=[];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%first residual
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%(imageD+1)-cone
Atmp=sparse(zeros(imageD+1,vars));
ctmp=sparse(zeros(imageD+1,1));

%radius
ctmp(1)=thr;

%coefficients

%f1=u11*1-h1'*U
%f2=u21*1-h2'*U

for ii=1:imageD,
    Atmp(ii+1,(sourceD+1)*ii:(sourceD+1)*ii+sourceD)=[Ut(:,1)',1];
end
ctmp(2:end)=ut(:,1);

if imageD==1,
    %a one-dimensional camera!!
    %make cone constraint into two inequality constraints:
    % |y|<=x    <=>   -x<=y<=x
    
    At_l=[At_l;Atmp(1,:)-Atmp(2,:);Atmp(1,:)+Atmp(2,:)];
    c_l=[c_l;ctmp(1)-ctmp(2);ctmp(1)+ctmp(2)];
    K.l=K.l+2;
else
    At=[At;Atmp];
    c=[c;ctmp];
    K.q=[K.q,imageD+1];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% residuals 2,3,...,nbrpoints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for cnt=1:nbrpoints-1,
    %indexing...
    indexa=Hvars+cnt;   %indexa=a_cnt

    %cone constraint for a_cnt

    Utmp=Ut(:,cnt+1); %source point
    utmp=ut(:,cnt+1); %image point

    %imageD+2-cone
    Atmp=sparse(zeros(imageD+2,vars));
    ctmp=sparse(zeros(imageD+2,1));

    %radius: lambda+a_cnt
    %lambda=p3*U
    
    Atmp(1,1:sourceD)=-Utmp';
    Atmp(1,indexa)=-1; %a_cnt
    ctmp(1)=1; %homogeneous one of Utmp;

    %coefficients

    %2*f1=2*(u11*lambda-h1'*U)
    %2*f2=2*(u21*lambda-h2'*U)
    
    Atmp(2:imageD+1,1:sourceD)=-2*utmp*Utmp';
    for ii=1:imageD,
        Atmp(1+ii,(sourceD+1)*ii:(sourceD+1)*ii+sourceD)=2*[Utmp',1];
    end
    ctmp(2:imageD+1)=2*utmp; %times homogeneous one of Utmp
    
    %h3'*U-a2
    Atmp(end,1:sourceD)=-Utmp';
    Atmp(end,indexa)=1;
    ctmp(end)=1; %homogeneous one of Utmp

    At=[At;Atmp];
    c=[c;ctmp];
    K.q=[K.q,imageD+2];
    
    %linear inequalities...
    %thr^2*lambda_i-alpha_i>=0
    Atmp=sparse(zeros(1,vars));
    ctmp=sparse(zeros(1,1));
    
    Atmp(1,1:sourceD)=-Utmp'*thr^2;
    Atmp(1,indexa)=1;
    ctmp(1)=thr^2; %homogeneous one of Utmp

    At_l=[At_l;Atmp];
    c_l=[c_l;ctmp];
    K.l=K.l+1;
    
    %depth should be >=rect_yL
    Atmp=sparse(zeros(1,vars));
    ctmp=sparse(zeros(1,1));
    Atmp(1,1:sourceD)=-Utmp';
    ctmp(1)=1-rect_yL(cnt); %homogeneous one of Utmp
    At_l=[At_l;Atmp];
    c_l=[c_l;ctmp];
    K.l=K.l+1;
    %depth should be <=rect_yU
    Atmp=sparse(zeros(1,vars));
    ctmp=sparse(zeros(1,1));
    Atmp(1,1:sourceD)=Utmp';
    ctmp(1)=rect_yU(cnt)-1; %homogeneous one of Utmp
    At_l=[At_l;Atmp];
    c_l=[c_l;ctmp];
    K.l=K.l+1;
end %cnt


b=sparse(zeros(vars,1));
pars=[];
pars.fid=0;

for ii=updateindex,
    Utmp=Ut(:,ii+1);
    b(1:sourceD)=Utmp;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SEDUMI

    [x,yU,infoU]=sedumi([At_l;At],b,[c_l;c],K,pars); %maximimize depth_i
    [x,yL,infoL]=sedumi([At_l;At],-b,[c_l;c],K,pars); %minimize depth_i

    bU=min(b'*real(yU)+1,rect_yU(ii));
    bL=max(b'*real(yL)+1,rect_yL(ii)); %homogeneous one of Utmp
    if infoU.numerr || infoU.dinf,
        'no upper bound'
    else
        rect_yU(ii)=bU;
    end
        
    if infoL.numerr || infoL.dinf,
        'no lower bound'
    else
        rect_yL(ii)=bL; %bounds on variables
    end
end


function hh=l2_reconstruct_loop(ut,Ut,rect_yL,rect_yU,xL,xU);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIMIZATION - over one region with convex envelope
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nbrpoints=size(ut,2);

sourceD=size(Ut,1);
imageD=size(ut,1);
nbrpoints=size(ut,2);

%variable order:
% last row [h_n1,h_n2,...,h_{n,n-1}] where it is assumed h_nn=1
% first row, second row, etc
% upper bounds a_1,...,a_n on numerators
% (nbrpoints-1)*3 convex envelope dummies (z,yp,zp)

Hvars= (sourceD+1)*(imageD+1)-1;
vars = Hvars + nbrpoints + 3*(nbrpoints-1);

index_z=Hvars+nbrpoints+[1:3:3*(nbrpoints-1)]; %location of z_2,z_3,...,z_nbrimages

feasible=1;

%sedumi matrices
At_l=sparse(zeros(0,vars)); %linear inequalities
c_l=sparse(zeros(0,1));
At=sparse(zeros(0,vars)); %cone constraints
c=sparse(zeros(0,1));
clear K;
K.l=0;
K.q=[];

b=sparse(zeros(vars,1));
b([Hvars+1,index_z])=-1; %minimize a_1,z_2,...,z_nbrimages

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%first residual
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%imageD+2-cone
Atmp=sparse(zeros(imageD+2,vars));
ctmp=sparse(zeros(imageD+2,1));

%radius
Atmp(1,Hvars+1)=-1;
ctmp(1)=1/4;

%coefficients

%f1=u11*1-h1'*U
%f2=u21*1-h2'*U

for ii=1:imageD,
    Atmp(ii+1,(sourceD+1)*ii:(sourceD+1)*ii+sourceD)=[Ut(:,1)',1];
end
ctmp(2:imageD+1)=ut(:,1);

Atmp(end,Hvars+1)=-1;
ctmp(end)=-1/4;

At=[At;Atmp];
c=[c;ctmp];
K.q=[K.q,imageD+2];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% residuals 2,3,...,nbrpoints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for cnt=1:nbrpoints-1,
    %indexing...
    index=index_z(cnt); %envolope variables start index=z,index+1=yp,index+2=zp
    indexa=Hvars+1+cnt;   %indexa=a_cnt

    %cone constraint for a_cnt

    Utmp=Ut(:,cnt+1); %source point
    utmp=ut(:,cnt+1); %image point

    %imageD+2-cone
    Atmp=sparse(zeros(imageD+2,vars));
    ctmp=sparse(zeros(imageD+2,1));

    %radius: lambda+a_cnt
    %lambda=p3*U
    
    Atmp(1,1:sourceD)=-Utmp';
    Atmp(1,indexa)=-1; %a_cnt
    ctmp(1)=1; %homogeneous one of Utmp;

    %coefficients

    %2*f1=2*(u11*lambda-h1'*U)
    %2*f2=2*(u21*lambda-h2'*U)
    
    Atmp(2:imageD+1,1:sourceD)=-2*utmp*Utmp';
    ctmp(2:imageD+1)=2*utmp; %homogeneous one of Utmp
    
    for ii=1:imageD,
        Atmp(ii+1,(sourceD+1)*ii:(sourceD+1)*ii+sourceD)=2*[Utmp',1];
    end

    %h3'*U-a2
    Atmp(end,1:sourceD)=-Utmp';
    Atmp(end,indexa)=1;
    ctmp(end)=1; %homogeneous one of Utmp

    At=[At;Atmp];
    c=[c;ctmp];
    K.q=[K.q,imageD+2];

    %convex envelope for positive quadrant of x/y
    if cnt<=sourceD,
        yL=rect_yL(cnt);
        yU=rect_yU(cnt);
    else
         %find upper and lower bounds for y
         LL=Ut(:,2:sourceD+1)';
         LL0=ones(sourceD,1);
         
         iLL=Utmp'*inv(LL);
         
         tmp1=iLL'.*rect_yL(1:sourceD);
         tmp2=iLL'.*rect_yU(1:sourceD);
         
         yU=sum(max(tmp1,tmp2))-iLL*LL0+1; %homogeneous one of Utmp
         yL=sum(min(tmp1,tmp2))-iLL*LL0+1; %homogeneous one of Utmp
         
         yU=min(yU,rect_yU(cnt));
         yL=max(yL,rect_yL(cnt));
         
         if yL>yU,
             feasible=0;
         end
    end

    %linear inequalities...
    Atmp=sparse(zeros(6,vars));
    ctmp=sparse(zeros(6,1));

    %RECALL: x means a_cnt and y means lambda
    %envolope variables start index=z,index+1=yp,index+2=zp

    %z-zp>=0
    Atmp(1,index)=-1;Atmp(1,index+2)=1;
    %zp>=0
    Atmp(2,index+2)=-1;
    %x-xL>=0 (a_cnt-xL>=0)
    Atmp(3,indexa)=-1;
    ctmp(3)=-xL;
    %xU-x>=0 (xU-a2>=0)
    Atmp(4,indexa)=1;
    ctmp(4)=xU;
    %y-yL>=0 (lambda-yL>=0),lambda=h3'*U
    
    Atmp(5,1:sourceD)=-Utmp';
    ctmp(5)=1-yL; %homogeneous one of Utmp
    
    %yU-y>=0 (yU-lambda>=0),lambda=h3'*U
    Atmp(6,1:sourceD)=Utmp';
    ctmp(6)=yU-1; %homogeneous one of Utmp

    At_l=[At_l;Atmp];
    c_l=[c_l;ctmp];
    K.l=K.l+6;


    %C and D-inequalities
    Atmp=sparse(zeros(4,vars));
    ctmp=sparse(zeros(4,1));

    %RECALL: x means a_cnt and y means lambda_cnt
    %envolope variables start index=z,index+1=yp,index+2=zp

    Atmp(1,index+1)=-1;
    Atmp(1,indexa)=-yL/(xU-xL);
    ctmp(1)=-yL*xU/(xU-xL);

    Atmp(2,index+1)=-1;
    Atmp(2,1:sourceD)=Utmp';
    Atmp(2,indexa)=-yU/(xU-xL);
    ctmp(2)=-yU*xL/(xU-xL)-1; %homogeneous one of Utmp

    Atmp(3,indexa)=yU/(xU-xL);
    Atmp(3,index+1)=1;
    ctmp(3)=yU*xU/(xU-xL);

    Atmp(4,1:sourceD)=-Utmp';
    Atmp(4,index+1)=1;
    Atmp(4,indexa)=yL/(xU-xL);
    ctmp(4)=yL*xL/(xU-xL)+1; %homogeneous one of Utmp

    At_l=[At_l;Atmp];
    c_l=[c_l;ctmp];
    K.l=K.l+4;

    %A-cone
    Atmp=sparse(zeros(3,vars));
    ctmp=sparse(zeros(3,1));

    %radius
    Atmp(1,index+1)=-(xU-xL)^2;
    Atmp(1,index+2)=-1;

    %coefficients
    Atmp(2,indexa)=2*sqrt(xL);
    ctmp(2)=2*sqrt(xL)*xU;

    Atmp(3,index+1)=-(xU-xL)^2;
    Atmp(3,index+2)=1;

    At=[At;Atmp];
    c=[c;ctmp];
    K.q=[K.q,3];

    %B-cone
    Atmp=sparse(zeros(3,vars));
    ctmp=sparse(zeros(3,1));

    %radius
    Atmp(1,1:sourceD)=-(xU-xL)^2*Utmp';
    Atmp(1,index+1)=(xU-xL)^2;
    Atmp(1,index)=-1;
    Atmp(1,index+2)=1;

    ctmp(1)=(xU-xL)^2; %homogeneous one of Utmp

    %coefficients
    Atmp(2,indexa)=-2*sqrt(xU);
    ctmp(2)=-2*sqrt(xU)*xL;

    Atmp(3,1:sourceD)=-(xU-xL)^2*Utmp';
    Atmp(3,index+1)=(xU-xL)^2;
    Atmp(3,index)=1;
    Atmp(3,index+2)=-1;

    ctmp(3)=(xU-xL)^2; %homogeneous one of Utmp

    At=[At;Atmp];
    c=[c;ctmp];
    K.q=[K.q,3];
    
end %cnt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SEDUMI

if feasible==1,
    pars=[];
    pars.fid=0;
    pars.eps=0;
    [x,y,info]=sedumi([At_l;At],b,[c_l;c],K,pars);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end of OPTIMIZATION - over one region
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if  feasible==0 || info.dinf==1, %no feasible solution
    res=Inf*ones(1,nbrpoints); %set residuals to infinity
    lowerbound=Inf;
else
    %feasible solution
    
    %create homography or projection
    H=ones(imageD+1,sourceD+1);
    for ii=1:imageD,
        H(ii,:)=y((sourceD+1)*ii:(sourceD+1)*ii+sourceD)';
    end
    H(end,1:end-1)=y(1:sourceD)';
    
    %compute residuals
    tmp=H*[Ut;ones(1,nbrpoints)];
    res=sum((ut-tmp(1:imageD,:)./(ones(imageD,1)*tmp(end,:))).^2);
    
    lowerbound=sum(y([Hvars+1,index_z]));
    hh.y=y;
    hh.H=H;
    
    %store depths
    hh.lambda=(H(end,:)*[Ut(:,2:end);ones(1,nbrpoints-1)])';
end

hh.res=res; %residuals
hh.lowerbound=lowerbound; %lowerbounds for each fractional term
