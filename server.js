const express = require("express");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

const API_URL = "http://160.250.247.143:9000/api";

// ====== CONFIG ======
const ADMIN = "@vanminh2603";
let history = []; // lưu 50 phiên

// ====== PHÂN TÍCH CẦU ======
function analyzePattern(data) {
    if (data.length < 5) return "random";

    const last = data.slice(-5).map(x => x.result);

    // cầu bệt
    if (last.every(x => x === "chan")) return "bet_chan";
    if (last.every(x => x === "le")) return "bet_le";

    // cầu 1-1
    let zigzag = true;
    for (let i = 1; i < last.length; i++) {
        if (last[i] === last[i - 1]) zigzag = false;
    }

    if (zigzag) {
        return last[last.length - 1] === "chan" ? "bet_le" : "bet_chan";
    }

    return "random";
}

// ====== AI DỰ ĐOÁN ======
function predict(history) {
    const pattern = analyzePattern(history);

    if (pattern === "bet_chan") return "chan";
    if (pattern === "bet_le") return "le";

    // fallback random nhẹ
    return Math.random() > 0.5 ? "chan" : "le";
}

// ====== CALL API ======
async function getData() {
    try {
        const res = await axios.get(API_URL, { timeout: 10000 });
        return res.data;
    } catch {
        return null;
    }
}

// ====== API CHÍNH ======
app.get("/api", async (req, res) => {
    const data = await getData();

    let phien = Date.now();
    let result = Math.random() > 0.5 ? "chan" : "le";

    if (data) {
        phien = data.phien || data.round || phien;
        result = data.ket_qua || result;
    }

    // lưu lịch sử
    history.push({ phien, result });
    if (history.length > 50) history.shift();

    const duDoan = predict(history);

    // fake xúc xắc
    const xucXac = [
        Math.random() > 0.5 ? "do" : "trang",
        Math.random() > 0.5 ? "do" : "trang",
        Math.random() > 0.5 ? "do" : "trang"
    ];

    const soDo = xucXac.filter(x => x === "do").length;
    const soTrang = 3 - soDo;

    res.json({
        success: true,
        admin: ADMIN,
        phien_hien_tai: phien,
        du_doan: duDoan,
        lich_su: history,
        pattern: analyzePattern(history),
        du_doan_xuc_xac: xucXac,
        cua_dat:
            soDo === 3 ? "3_do" :
            soTrang === 3 ? "3_trang" :
            soDo === 2 ? "2_do_1_trang" :
            "1_do_2_trang",
        so_do: soDo,
        so_trang: soTrang
    });
});

// ====== ROUTE CHECK ======
app.get("/", (req, res) => {
    res.send("🚀 API SOI CẦU VIP RUNNING - ADM " + ADMIN);
});

// ====== START ======
app.listen(PORT, () => {
    console.log("🔥 Server chạy tại port " + PORT);
});
