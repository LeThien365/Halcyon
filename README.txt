HALCYON - MA NGUON DAY DU
=========================

CAU TRUC
  include/, src/        Engine (build thanh halcyon / halcyon.exe)
  build.sh, build.bat   Lenh build san
  tb_data/              5 bang tablebase 3 quan (dinh dang rieng cua Halcyon)
  eco_data/             Bo khai cuoc ECO A00-E99 (Lichess, 3864 dong)
  tools/nnue_convert.cpp   Mo rong mang 768 -> HalfKAv2 (build RIENG, co main() rieng; chi cho mang 256)
  tools/tb/             Sinh/kiem tra tablebase (build RIENG)
  tools/train/          Script Python: self-play, cham diem, PGN, train, danh gia khai cuoc, engine_match.py
  tools/tests/t_eval.cpp  Kiem tra doi xung mau cua eval (build RIENG)
  backup/               eval.cpp / eval.h TRUOC khi nang cap (chep de len src/ include/ neu muon quay lai)

BUILD (chi include/ + src/, KHONG dua file trong tools/ vao lenh nay)
  Windows : build.bat
  Linux   : ./build.sh            (hidden 1024)   |  ./build.sh 256   (dung mang 256 nhung san trong engine)

CO BUILD
  -DHALCYON_HIDDEN_SIZE=N   Do rong lop an. Mac dinh 1024. File .hnn phai train DUNG do rong nay
                            (header .hnn mang so do; sai do rong thi engine tu choi nap).
  -DEVAL_DISABLE_MASK=bits  Tat tung phan nang cap eval de A/B:
                            1 mobility an toan | 2 tot thong bi chan/di tu do | 4 tot thong khong the can
                            8 tan cong vua can hau | 16 scale the hoa | 32 giam diem theo luat 50 nuoc

LUU Y QUAN TRONG VE 1024
  - Mang compiled-in trong engine (nnue_default_weights.h) rong 256. Ban build 1024 KHONG dung duoc no:
    khong nap mang thi engine tu choi NNUE va dung HCE. Hay nap mang cua ban:
        setoption name NNUEFile value halcyon_net.hnn
    Engine se in: do rong, can tren int16 accumulator, can tren int32 dau ra, va "arithmetic proven safe"
    hoac "CANNOT PROVE SAFE". Can tren la bao thu: khong phai ket luan tran so, nhung nen xem xet.
  - Python trainer (tools/train/nnue_format.py) o day van cau hinh 256. Ban train 1024 can sua HIDDEN_SIZE
    o nnue_format.py va cac script train cho khop.

UCI: Hash, Threads, NNUEFile, BookFile, OwnBook, UseNNUE, MultiPV, SyzygyPath (= thu muc tb_data, dinh dang rieng cua Halcyon)

NANG CAP EVAL (eval.cpp / eval.h)
  evaluate()      tablebase (<=3 quan) -> NNUE hoac HCE -> scale the hoa ly thuyet -> giam diem theo luat 50 nuoc
  evaluate_hce()  chi phan thu cong (dung de sinh nhan, so sanh)
  Do bang tran dau (ban moi vs ban cu, co dinh depth, khai cuoc ECO doi mau):
    HCE, depth 5, 1000 van : 53.4%  (+24 Elo, 95% CI +5..+43)
    HCE, depth 7,  300 van : 53.7%  (+26 Elo, 95% CI -6..+57)
    Duong NNUE, depth 5, 800 van: 50.7% (+5 Elo, CI -17..+26)  -> KHONG do duoc khac biet
  Luat the hoa sua loi ro rang: Hai Ma vs Vua, Xe vs Tuong khong con tot, Xe+Tuong vs Xe... truoc day duoc cham +300..+600.
