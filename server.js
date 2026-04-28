const express = require("express");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

const URL_TX = "https://wtx.tele68.com/v1/tx/sessions";
const URL_MD5 = "https://wtxmd52.tele68.com/v1/txmd5/sessions";

const http = axios.create({ timeout: 10000 });

// ================= MARKOV =================
class Markov {
  constructor(order = 3){
    this.order = order;
    this.map = {};
    this.h = [];
  }

  train(seq){
    this.h = seq;
    this.map = {};
    for(let i=this.order;i<seq.length;i++){
      const key = seq.slice(i-this.order,i).join("|");
      const cur = seq[i];
      if(!this.map[key]) this.map[key]={};
      this.map[key][cur] = (this.map[key][cur]||0)+1;
    }
  }

  predict(){
    const key = this.h.slice(-this.order).join("|");
    const m = this.map[key];
    if(!m) return {state:"TAI", prob:0.5};

    let total=0, best=null, max=0;
    for(let k in m){
      total+=m[k];
      if(m[k]>max){max=m[k];best=k;}
    }
    return {state:best, prob:max/total};
  }
}

// ================= EMA =================
class EMA{
  constructor(a=0.3){this.a=a;this.v=null;}
  train(arr){
    if(!arr.length)return;
    this.v = arr[0];
    for(let i=1;i<arr.length;i++){
      this.v = this.a*arr[i] + (1-this.a)*this.v;
    }
  }
  predict(){return this.v||10.5;}
}

// ================= PATTERN =================
class Pattern{
  constructor(){this.h=[];}
  train(seq){this.h=seq;}

  predict(){
    const l=this.h;
    if(l.length<3) return {o:"TAI",t:"thiếu"};

    const last=l[l.length-1];

    // bệt
    let streak=1;
    for(let i=l.length-2;i>=0;i--){
      if(l[i]===last) streak++;
      else break;
    }
    if(streak>=4) return {o:last,t:"bệt"};

    return {o:last,t:"xu hướng"};
  }
}

// ================= AI =================
class AI{
  constructor(name){
    this.name=name;
    this.h=[];

    this.markov=new Markov(3);
    this.ema=new EMA(0.3);
    this.pattern=new Pattern();

    // học trọng số
    this.weights={
      m:1,
      ema:1,
      pat:1,
      bridge:1
    };

    this.historyPredict=[];
  }

  add(data){
    const ids=new Set(this.h.map(x=>x.id));

    data.forEach(s=>{
      if(!ids.has(s.id) && s.dices){
        const sum=s.dices.reduce((a,b)=>a+b,0);
        this.h.push({
          id:s.id,
          sum,
          o:sum>=11?"TAI":"XIU"
        });
      }
    });

    if(this.h.length>120)
      this.h=this.h.slice(-120);

    this.update();
    this.learn();
  }

  update(){
    const o=this.h.map(x=>x.o);
    const s=this.h.map(x=>x.sum);

    this.markov.train(o);
    this.ema.train(s);
    this.pattern.train(o);
  }

  // ===== cầu đảo / gãy =====
  bridge(){
    const s=this.h.map(x=>x.o);
    const l=s.length;
    if(l<6) return {type:"-",pred:"TAI"};

    // đảo
    let alt=true;
    for(let i=l-1;i>l-5;i--){
      if(s[i]===s[i-1]) alt=false;
    }
    if(alt){
      return {type:"đảo",pred:s[l-2]};
    }

    // gãy
    if(s[l-1]!==s[l-2]){
      return {type:"gãy",pred:s[l-2]};
    }

    return {type:"bt",pred:s[l-1]};
  }

  // ===== predict =====
  predict(){
    if(this.h.length<5)
      return {dd:"TAI",conf:50};

    const m=this.markov.predict();
    const ema=this.ema.predict();
    const pat=this.pattern.predict();
    const br=this.bridge();

    let p=0;

    // markov
    p += (m.state==="TAI"?1:0) * this.weights.m;

    // ema
    p += (ema>10.5?1:0) * this.weights.ema;

    // pattern
    p += (pat.o==="TAI"?1:0) * this.weights.pat;

    // bridge
    p += (br.pred==="TAI"?1:0) * this.weights.bridge;

    let totalW = this.weights.m + this.weights.ema + this.weights.pat + this.weights.bridge;

    let prob = p / totalW;

    const dd = prob>0.5?"TAI":"XIU";
    const conf = Math.round(Math.abs(prob-0.5)*200);

    // chia vốn
    let von="BỎ", lvl="RISK";
    if(conf>=75){von="3-5%";lvl="SAFE";}
    else if(conf>=60){von="2%";lvl="MID";}
    else if(conf>=52){von="1%";lvl="NHẸ";}

    const nextId = this.h[this.h.length-1].id+1;

    // lưu để học
    this.historyPredict.push({
      id:nextId,
      dd,
      real:null,
      m:m.state,
      ema:ema>10.5?"TAI":"XIU",
      pat:pat.o,
      br:br.pred
    });

    return {dd,conf,von,lvl,br};
  }

  // ===== học =====
  learn(){
    for(let p of this.historyPredict){
      const real = this.h.find(x=>x.id===p.id);
      if(!real) continue;

      p.real = real.o;

      const lr=0.1;

      if(p.m===p.real) this.weights.m+=lr; else this.weights.m-=lr;
      if(p.ema===p.real) this.weights.ema+=lr; else this.weights.ema-=lr;
      if(p.pat===p.real) this.weights.pat+=lr; else this.weights.pat-=lr;
      if(p.br===p.real) this.weights.bridge+=lr; else this.weights.bridge-=lr;

      // clamp
      for(let k in this.weights){
        this.weights[k]=Math.max(0.1,Math.min(2,this.weights[k]));
      }
    }

    this.historyPredict = this.historyPredict.slice(-20);
  }

  analysis(){
    if(!this.h.length) return {};

    const last=this.h[this.h.length-1];
    const p=this.predict();

    return {
      phien:last.id,
      kq:last.o,
      tong:last.sum,
      next:last.id+1,

      dd:p.dd,
      conf:p.conf,
      goi_y:p.von,
      lvl:p.lvl,

      cau:p.br.type,
      color:p.dd==="TAI"?"green":"red",

      note:p.conf<60?"Không chắc → đánh nhẹ":"Có thể vào",
      by:"@vanminh2603"
    };
  }
}

// ================= INIT =================
const normal=new AI("TX");
const md5=new AI("MD5");

// ================= FETCH =================
async function poll(){
  try{
    const [a,b]=await Promise.all([
      http.get(URL_TX),
      http.get(URL_MD5)
    ]);

    normal.add(a.data.list||[]);
    md5.add(b.data.list||[]);

    console.log("OK");
  }catch(e){
    console.log("ERR");
  }
}

setInterval(poll,5000);
poll();

// ================= API =================
app.get("/",(req,res)=>{
  res.json({
    api:["/taixiu","/taixiumd5","/all"],
    by:"@vanminh2603"
  });
});

app.get("/taixiu",(req,res)=>res.json(normal.analysis()));
app.get("/taixiumd5",(req,res)=>res.json(md5.analysis()));

app.get("/all",(req,res)=>{
  res.json({
    taixiu:normal.analysis(),
    md5:md5.analysis(),
    by:"@vanminh2603"
  });
});

// ================= START =================
app.listen(PORT,()=>console.log("RUN",PORT));
