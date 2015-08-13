function [prec,rec,F1,acc]=calc_measures(pred_label,labs)

gts=labs;
pred=pred_label==1;
npred=pred_label~=1;
true_IX=find(pred==gts);
false_IX=find(pred~=gts);
tp=sum(pred(true_IX));
fp=sum(pred(false_IX));
tn=sum(npred(true_IX));
fn=sum(npred(false_IX));

prec=tp / ((tp+fp)+eps);
rec=tp / ((tp + fn)+eps);
F1=(2*tp)/((2*tp+fp+fn)+eps);
acc=(tp+tn)/((tp+fp+tn+fn)+eps);
end

