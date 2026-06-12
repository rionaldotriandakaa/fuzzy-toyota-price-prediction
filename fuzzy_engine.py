import numpy as np

# ============================================================
# FUNGSI FUZZY MEMBERSHIP (VERSI ROBUST)
# ============================================================
def trapmf(x, a, b, c, d):
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        # Penanganan titik batas agar tidak 0
        if (a == b and xi <= a) or (c == d and xi >= c):
            y[i] = 1.0
        elif xi <= a or xi >= d:
            y[i] = 0.0
        elif b <= xi <= c:
            y[i] = 1.0
        elif a < xi < b:
            y[i] = (xi - a) / (b - a) if (b - a) != 0 else 1.0
        elif c < xi < d:
            y[i] = (d - xi) / (d - c) if (d - c) != 0 else 1.0
    return y if len(y) > 1 else float(y[0])

def trimf(x, a, b, c):
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= a or xi >= c:
            y[i] = 0.0
        elif xi == b:
            y[i] = 1.0
        elif a < xi <= b:
            y[i] = (xi - a) / (b - a) if (b - a) != 0 else 1.0
        elif b < xi < c:
            y[i] = (c - xi) / (c - b) if (c - b) != 0 else 1.0
    return y if len(y) > 1 else float(y[0])

def interp_mf(universe, mf_vals, x):
    return float(np.interp(x, universe, mf_vals))

# Universe
U_YEAR       = np.arange(1998, 2021, 1)
U_MILEAGE    = np.arange(0, 281000, 500)
U_TAX        = np.arange(0, 11_000_000, 10000)
U_MPG        = np.arange(2.8, 236.0, 0.2)
U_ENGINESIZE = np.arange(0.0, 4.6, 0.1)
U_PRICE      = np.arange(15_300_000, 1_090_000_000, 10000)

MF = {
    'year_Lama'      : trapmf(U_YEAR,       1998,1998,2016,2017),
    'year_Sedang'    : trimf( U_YEAR,       2016,2017,2018),
    'year_Baru'      : trapmf(U_YEAR,       2017,2018,2020,2020),
    'mil_Rendah'     : trapmf(U_MILEAGE,    0,0,15202,29794),
    'mil_Sedang'     : trimf( U_MILEAGE,    15202,29794,49992),
    'mil_Tinggi'     : trapmf(U_MILEAGE,    29794,49992,280699,280699),
    'tax_Murah'      : trapmf(U_TAX,        0,0,0,2_430_000),
    'tax_Normal'     : trimf( U_TAX,        0,2_430_000,2_610_000),
    'tax_Mahal'      : trapmf(U_TAX,        2_430_000,2_610_000,10_170_000,10_170_000),
    'mpg_Boros'      : trapmf(U_MPG,        2.8,2.8,55.4,62.8),
    'mpg_Irit'       : trimf( U_MPG,        55.4,62.8,69.0),
    'mpg_SangatIrit' : trapmf(U_MPG,        62.8,69.0,235.0,235.0),
    'eng_Kecil'      : trapmf(U_ENGINESIZE, 0.0,0.0,1.0,1.5),
    'eng_Sedang'     : trimf( U_ENGINESIZE, 1.0,1.5,1.8),
    'eng_Besar'      : trapmf(U_ENGINESIZE, 1.5,1.8,4.5,4.5),
    'price_Murah'    : trapmf(U_PRICE, 15_300_000,15_300_000,149_220_000,194_310_000),
    'price_Sedang'   : trimf( U_PRICE, 149_220_000,194_310_000,269_910_000),
    'price_Mahal'    : trapmf(U_PRICE, 194_310_000,269_910_000,1_079_910_000,1_079_910_000),
}

ATURAN = [
    (['year_Baru','mil_Rendah'],                  'Mahal'),
    (['year_Baru','mil_Sedang'],                  'Mahal'),
    (['year_Baru','tax_Mahal'],                   'Mahal'),
    (['year_Baru','eng_Besar'],                   'Mahal'),
    (['eng_Besar','tax_Mahal'],                   'Mahal'),
    (['year_Baru','mil_Rendah','tax_Mahal'],      'Mahal'),
    (['year_Baru','mil_Rendah','mpg_SangatIrit'], 'Mahal'),
    (['year_Lama','mil_Tinggi'],                  'Murah'),
    (['year_Lama','mil_Sedang'],                  'Murah'),
    (['year_Lama','tax_Murah'],                   'Murah'),
    (['year_Lama','eng_Kecil'],                   'Murah'),
    (['year_Lama','mpg_Boros'],                   'Murah'),
    (['eng_Kecil','tax_Murah'],                   'Murah'),
    (['year_Lama','mil_Tinggi','tax_Murah'],      'Murah'),
    (['year_Lama','mil_Tinggi','mpg_Boros'],      'Murah'),
    (['year_Sedang','mil_Sedang'],                'Sedang'),
    (['year_Sedang','tax_Normal'],                'Sedang'),
    (['year_Sedang','eng_Sedang'],                'Sedang'),
    (['year_Sedang','mpg_Irit'],                  'Sedang'),
    (['year_Sedang','mil_Sedang','tax_Normal'],   'Sedang'),
    (['year_Baru','mil_Tinggi'],                  'Sedang'),
    (['year_Lama','mil_Rendah'],                  'Sedang'),
    (['eng_Besar','mpg_SangatIrit'],              'Sedang'),
    (['eng_Kecil','year_Baru'],                   'Sedang'),
    (['year_Sedang','mil_Rendah','tax_Murah'],    'Sedang'),
]

# ← SAMA PERSIS dengan notebook
SINGLETON = {
    'Murah' : 100_000_000,
    'Sedang': 210_000_000,
    'Mahal' : 550_000_000,
}

def fuzzifikasi(yr, mi, tx, mpgv, en):
    # Clamp dulu
    yr   = int(np.clip(yr,    1998, 2020))
    mi   = float(np.clip(mi,  0,    280699))
    tx   = float(np.clip(tx,  0,    10_170_000))
    mpgv = float(np.clip(mpgv,2.8,  235.0))
    en   = float(np.clip(en,  0.0,  4.5))
    return {
        'year_Lama'      : interp_mf(U_YEAR,       MF['year_Lama'],      yr),
        'year_Sedang'    : interp_mf(U_YEAR,       MF['year_Sedang'],    yr),
        'year_Baru'      : interp_mf(U_YEAR,       MF['year_Baru'],      yr),
        'mil_Rendah'     : interp_mf(U_MILEAGE,    MF['mil_Rendah'],     mi),
        'mil_Sedang'     : interp_mf(U_MILEAGE,    MF['mil_Sedang'],     mi),
        'mil_Tinggi'     : interp_mf(U_MILEAGE,    MF['mil_Tinggi'],     mi),
        'tax_Murah'      : interp_mf(U_TAX,        MF['tax_Murah'],      tx),
        'tax_Normal'     : interp_mf(U_TAX,        MF['tax_Normal'],     tx),
        'tax_Mahal'      : interp_mf(U_TAX,        MF['tax_Mahal'],      tx),
        'mpg_Boros'      : interp_mf(U_MPG,        MF['mpg_Boros'],      mpgv),
        'mpg_Irit'       : interp_mf(U_MPG,        MF['mpg_Irit'],       mpgv),
        'mpg_SangatIrit' : interp_mf(U_MPG,        MF['mpg_SangatIrit'], mpgv),
        'eng_Kecil'      : interp_mf(U_ENGINESIZE, MF['eng_Kecil'],      en),
        'eng_Sedang'     : interp_mf(U_ENGINESIZE, MF['eng_Sedang'],     en),
        'eng_Besar'      : interp_mf(U_ENGINESIZE, MF['eng_Besar'],      en),
    }

def mamdani(mu_dict):
    """Agregasi SUM + defuzzifikasi centroid"""
    agg = np.zeros(len(U_PRICE))
    for kondisi, output in ATURAN:
        firing  = min(mu_dict[k] for k in kondisi)
        if firing <= 0:
            continue
        clipped = np.minimum(firing, MF[f'price_{output}'])
        agg = np.maximum(agg, clipped)
    total_area = np.trapz(agg, U_PRICE)
    if total_area == 0:
        return 0.0
    return float(np.clip(
        np.trapz(agg * U_PRICE, U_PRICE) / total_area,
        15_300_000, 1_079_910_000
    ))

def sugeno(mu_dict):
    num, den = 0.0, 0.0
    for kondisi, output in ATURAN:
        firing = min(mu_dict[k] for k in kondisi)
        num   += firing * SINGLETON[output]
        den   += firing
    return float(np.clip(
        num / den if den > 0 else 0.0,
        15_300_000, 1_079_910_000
    ))

def klasifikasi(harga):
    if harga < 194_310_000:   return 'Murah'
    elif harga < 269_910_000: return 'Sedang'
    else:                      return 'Mahal'

def prediksi_lengkap(yr, mi, tx, mpgv, en):
    mu = fuzzifikasi(yr, mi, tx, mpgv, en)
    hm = mamdani(mu)
    hs = sugeno(mu)
    return {
        'mu'     : mu,
        'mamdani': hm,
        'sugeno' : hs,
        'kls_md' : klasifikasi(hm),
        'kls_sg' : klasifikasi(hs),
        'selisih': abs(hm - hs),
        'persen' : abs(hm-hs)/hm*100 if hm > 0 else 0,
    }