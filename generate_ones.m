function [x,t] =generate_ones(N)
% N=20;
x = randi([0 1],N,1);
p=0;
pp=0;
t=[];
t(1)=0;
% t(2)=0;
for i=2:N
    c=x(i);
    p=x(i-1);
%     pp=x(i-2);
%     if p && pp && c
 if p &&  c
        t(i)=1;
    else
        t(i)=0;
    end
end
t=t';
