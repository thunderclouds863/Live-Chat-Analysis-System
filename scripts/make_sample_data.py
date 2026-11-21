"""Generate synthetic sample files for the demo (no real customer data).

Usage: python scripts/make_sample_data.py  -> sample_data/*.xlsx
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "sample_data"
LEAVE = ("Mohon maaf, dikarenakan tidak ada respon, chat ini Kami akhiri. Terima kasih telah "
         "menggunakan layanan Live Chat Toyota Astra Motor, selamat beraktivitas kembali.")

NORMAL = [
    ("Berapa harga Avanza tipe G terbaru untuk wilayah Jakarta?",
     "Untuk informasi harga Avanza tipe G wilayah Jakarta saat ini mulai Rp 260 juta, silakan menghubungi dealer terdekat untuk penawaran resmi."),
    ("Bagaimana cara booking service berkala lewat aplikasi?",
     "Caranya buka aplikasi mToyota, pilih menu Booking Service, lalu pilih dealer dan jadwal yang tersedia."),
    ("Apakah garansi mesin masih berlaku kalau servis di bengkel umum?",
     "Garansi tetap berlaku selama servis sesuai prosedur dan menggunakan suku cadang asli, informasi lengkap ada di buku garansi."),
]
SERIOUS = [
    ("Mobil saya baru 2 bulan tapi AC tidak dingin sama sekali, sudah ke dealer belum ada solusi",
     "Baik Bapak, keluhan ini akan kami bantu koordinasikan dengan tim terkait dan dealer untuk dilakukan pengecekan.",
     "Bapak, dealer sudah dikoordinasikan dan akan menghubungi Bapak besok untuk jadwal perbaikan kompresor AC."),
    ("Sudah 3 minggu suku cadang pesanan saya belum datang, tolong dicek",
     "Mohon maaf atas ketidaknyamanannya, akan kami teruskan ke tim terkait untuk dicek status pesanan.",
     "Status pesanan sudah dicek, suku cadang dijadwalkan tiba di dealer hari Jumat dan dealer akan menghubungi Bapak."),
]


def build():
    rnd = random.Random(7)
    rows, complaints = [], []
    base = datetime(2026, 10, 1, 8, 0)
    no = 1
    for t in range(1, 41):
        ticket = f"TKT-OCT-{t:03d}"
        phone = f"0812{rnd.randint(10000000, 99999999)}"
        start = base + timedelta(days=rnd.randint(0, 28), hours=rnd.randint(0, 9), minutes=rnd.randint(0, 59))
        kind = rnd.choices(["normal", "serious", "complaint", "abandoned"], [5, 2, 1, 1])[0]
        msgs = [("Bot", "System", "Selamat datang di layanan Live Chat Toyota Astra Motor. Mohon tuliskan nama dan nomor handphone.", 0),
                ("Customer", f"Customer {t}", f"Nama saya Pelanggan {t}, nomor {phone}", 1)]
        greet = rnd.choice(["Selamat pagi", "Selamat siang", "Selamat sore"])
        msgs.append(("Operator", "Agent Demo", f"{greet}, dengan Agent Demo, ada yang bisa dibantu?", rnd.randint(2, 4)))
        if kind == "normal":
            q, a = rnd.choice(NORMAL)
            msgs += [("Customer", f"Customer {t}", q, 5), ("Operator", "Agent Demo", a, 5 + rnd.randint(1, 9)),
                     ("Operator", "Agent Demo", "Apakah informasinya sudah cukup? Terima kasih telah menghubungi kami.", 20)]
        elif kind in ("serious", "complaint"):
            q, a1, a2 = rnd.choice(SERIOUS)
            final_gap = rnd.randint(60, 400) if kind == "serious" else rnd.randint(1500, 6000)
            msgs += [("Customer", f"Customer {t}", q, 5), ("Operator", "Agent Demo", a1, 5 + rnd.randint(1, 8)),
                     ("Ticket Automation", "System", "Ticket Has Been Reopened by Agent Demo", final_gap - 1),
                     ("Operator", "Agent Demo", a2, final_gap)]
            if kind == "complaint":
                complaints.append({"No": len(complaints) + 1, "No.Handphone": phone,
                                   "Nama": f"Pelanggan {t}", "Kategori": "Produk",
                                   "Lead Time (Solved)": round(final_gap / 1440, 1)})
        else:
            msgs += [("Customer", f"Customer {t}", "mau tanya", 5), ("Ticket Automation", "System", LEAVE, 40)]
        for role, sender, text, minute in msgs:
            rows.append({"No": no, "Ticket Number": ticket, "Role": role, "Sender": sender,
                         "Message Date": (start + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S"),
                         "Message": text})
            no += 1
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_excel(OUT / "raw_conversation_sample.xlsx", index=False)
    pd.DataFrame(complaints).to_excel(OUT / "complaint_sample.xlsx", index=False)
    print(len(rows), "messages,", len(complaints), "complaints")


if __name__ == "__main__":
    build()
