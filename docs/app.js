'use strict';

const svgNS = 'http://www.w3.org/2000/svg';
const plot = document.getElementById('decision-plot');
const priceButtons = Array.from(document.querySelectorAll('[data-price]'));
const colors = {gain_only:'#8D9FCD',cost_only:'#A9BACF',gain_and_cost:'#62BE9F'};
const labels = {gain_only:'Gain information',cost_only:'Cost information',gain_and_cost:'Gain + cost information'};
let measurements;

function svgElement(name, attrs, content) {
  const el = document.createElementNS(svgNS,name);
  for (const [key,value] of Object.entries(attrs)) el.setAttribute(key,String(value));
  if (content !== undefined) el.textContent=content;
  return el;
}
function signed(value, digits=2) {
  return `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(digits)}`;
}
function renderPrice(price) {
  if (!measurements) return;
  const row=measurements.prices[price];
  const compact=window.innerWidth<600;
  const min=-2, max=6.5, left=compact?132:206, right=compact?381:604;
  plot.setAttribute('viewBox',compact?'0 0 430 240':'0 0 650 240');
  const x=v=>left+(v-min)/(max-min)*(right-left);
  plot.replaceChildren();
  const title=svgElement('title',{},`Decision utility at token valuation ${price}`);
  const desc=svgElement('desc',{},'Utility differences from the best constant action, in percentage points. Whiskers show 95% confidence intervals. All selection observations are free.');
  plot.append(title,desc);
  for (const tick of [-2,0,2,4,6]) {
    plot.append(svgElement('line',{x1:x(tick),x2:x(tick),y1:18,y2:190,stroke:tick===0?'#AAB8C4':'#E8EDF1','stroke-width':tick===0?1.4:1,'stroke-dasharray':tick===0?'4 4':'none'}));
    plot.append(svgElement('text',{x:x(tick),y:213,fill:'#657489','font-size':11,'text-anchor':'middle'},tick));
  }
  const keys=['gain_only','cost_only','gain_and_cost'];
  keys.forEach((key,index)=>{
    const item=row.rules[key], y=49+index*58;
    const value=item.mean*100, lo=item.ci[0]*100, hi=item.ci[1]*100;
    plot.append(svgElement('text',{x:2,y:y+4,fill:'#263747','font-size':13,'font-weight':key==='gain_and_cost'?700:400},compact?{gain_only:'Gain information',cost_only:'Cost information',gain_and_cost:'Gain + cost'}[key]:labels[key]));
    plot.append(svgElement('line',{x1:x(lo),x2:x(hi),y1:y,y2:y,stroke:colors[key],'stroke-width':3,'stroke-linecap':'round'}));
    for (const end of [lo,hi]) plot.append(svgElement('line',{x1:x(end),x2:x(end),y1:y-6,y2:y+6,stroke:colors[key],'stroke-width':1.5}));
    plot.append(svgElement('circle',{cx:x(value),cy:y,r:5.5,fill:colors[key],stroke:'white','stroke-width':2}));
    plot.append(svgElement('text',{x:compact?427:641,y:y+4,fill:'#263747','font-size':12,'text-anchor':'end','font-weight':key==='gain_and_cost'?700:400},signed(value)));
  });
  plot.append(svgElement('text',{x:(left+right)/2,y:237,fill:'#657489','font-size':10,'text-anchor':'middle'},'Utility difference (percentage points)'));
  document.getElementById('best-constant').textContent=row.constant;
  document.getElementById('decision-value').replaceChildren(document.createTextNode(signed(row.rules.gain_and_cost.mean*100)),Object.assign(document.createElement('span'),{textContent:'pp'}));
  document.getElementById('price-description').textContent=Number(price)===0?'At zero token cost, the benchmark compares answer correctness.':`At ${price}, each 1,000 tokens costs ${Number(price)*100} utility percentage ${Number(price)*100===1?'point':'points'}.`;
  priceButtons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.price===price)));
}
priceButtons.forEach(button=>button.addEventListener('click',()=>renderPrice(button.dataset.price)));
fetch('assets/results.json').then(response=>{
  if(!response.ok) throw new Error('The recorded measurements could not be loaded.');
  return response.json();
}).then(data=>{measurements=data;renderPrice('0.005');}).catch(()=>{
  plot.replaceChildren(svgElement('text',{x:12,y:90,fill:'#657489','font-size':14},'Open the downloadable measurements to inspect all five prices.'));
});

document.querySelectorAll('[data-copy]').forEach(button=>{
  button.addEventListener('click',async()=>{
    const target=document.getElementById(button.dataset.copy);
    const value=target.textContent;
    let copied=false;
    if (navigator.clipboard && window.isSecureContext) {
      try {await navigator.clipboard.writeText(value);copied=true;} catch {}
    }
    if (!copied) {
      const field=document.createElement('textarea');
      field.value=value;field.style.position='fixed';field.style.opacity='0';
      document.body.appendChild(field);field.select();
      copied=document.execCommand('copy');field.remove();
    }
    if (copied) {
      const previous=button.textContent;button.textContent='Copied';
      window.setTimeout(()=>{button.textContent=previous;},1800);
    } else {
      const selection=window.getSelection();const range=document.createRange();
      range.selectNodeContents(target);selection.removeAllRanges();selection.addRange(range);
      button.textContent='Text selected';
    }
  });
});

let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>renderPrice(document.querySelector('[data-price][aria-pressed="true"]').dataset.price),100);});
