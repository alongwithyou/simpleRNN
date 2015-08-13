function [x,t] = generate_sine(N,nr,fs1,fs2,fs3)
% N=1024;
% % fs=200;
% fs1=100;
% fs2=200;

% if nargin<5
%     nr=0.01;
% end

f=1;
ts1=1/fs1;
t1 = ts1*(0:N-1);
x1=(sin(2*pi*f*t1))./3;
x1=ones(size(x1))./2+x1;

if nargin==3
    x=x1;
    t=t1;
elseif nargin==4
    ts2=1/fs2;
    t2 = ts2*(0:N-1);
    x2=(sin(2*pi*f*t2))./3;
    x2=ones(size(x2))./2+x2;
    
    x=x1+x2;
    t=t1;
    x=x./2;
elseif nargin==5
    ts2=1/fs2;
    t2 = ts2*(0:N-1);
    x2=(sin(2*pi*f*t2))./3;
    x2=ones(size(x2))./2+x2;
    
    ts3=1/fs3;
    t3 = ts2*(0:N-1);
    x3=(sin(2*pi*f*t3))./3;
    x3=ones(size(x3))./2+x3;

    x=x1+x2+x3;
    t=t1;
    x=x./3;
end
x=x+randn(size(x,1),N)*nr;
% plot(t1,x)

