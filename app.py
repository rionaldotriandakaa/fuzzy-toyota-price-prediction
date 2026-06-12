import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzy_engine import (
    fuzzifikasi, mamdani, sugeno, klasifikasi,
    prediksi_lengkap, MF, U_YEAR, U_MILEAGE,
    U_TAX, U_MPG, U_ENGINESIZE, U_PRICE, ATURAN
)

# ============================================================
#  KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(page_title='Estimasi Harga Toyota Bekas', page_icon='🚗', layout='wide')

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #0D47A1; text-align: center; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1rem; color: #546E7A; text-align: center; margin-bottom: 1.5rem; }
    .result-card { background: #E3F2FD; border-radius: 12px; padding: 1.2rem; border-left: 5px solid #1565C0; margin-bottom: 1rem; }
    .result-card-orange { background: #FFF3E0; border-radius: 12px; padding: 1.2rem; border-left: 5px solid #E65100; margin-bottom: 1rem; }
    .badge-murah  { background:#C8E6C9; color:#1B5E20; padding:3px 12px; border-radius:20px; font-weight:600; font-size:0.9rem; }
    .badge-sedang { background:#FFF9C4; color:#F57F17; padding:3px 12px; border-radius:20px; font-weight:600; font-size:0.9rem; }
    .badge-mahal  { background:#FFCDD2; color:#B71C1C; padding:3px 12px; border-radius:20px; font-weight:600; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚗 Estimasi Harga Toyota Bekas</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Inferensi Fuzzy Mamdani & Sugeno · Kelompok My Little Bolu Ketan · IF-48-06</p>', unsafe_allow_html=True)
st.divider()

# ============================================================
#  SIDEBAR — INPUT
# ============================================================

st.sidebar.header('🔧 Input Spesifikasi Mobil')
st.sidebar.markdown('Atur nilai sesuai mobil yang ingin diestimasi:')

INPUT_YEAR       = st.sidebar.slider('Tahun Pembuatan', 1998, 2020, 2018, step=1)
INPUT_MILEAGE    = st.sidebar.number_input('Jarak Tempuh (km)', min_value=0, max_value=280699, value=25000, step=500)
INPUT_TAX        = st.sidebar.number_input('Pajak Tahunan (Rp)', min_value=0, max_value=10_170_000, value=2_500_000, step=10000)
INPUT_MPG        = st.sidebar.slider('Konsumsi BBM (mpg)', 2.8, 235.0, 65.0, step=0.1)
INPUT_ENGINESIZE = st.sidebar.slider('Kapasitas Mesin (L)', 0.0, 4.5, 1.5, step=0.1)

st.sidebar.divider()
hitung = st.sidebar.button('🔍 Estimasi Harga', use_container_width=True)

# ============================================================
#  HITUNG OTOMATIS
# ============================================================

yr   = int(np.clip(INPUT_YEAR, 1998, 2020))
mi   = float(np.clip(INPUT_MILEAGE, 0, 280699))
tx   = float(np.clip(INPUT_TAX, 0, 10_170_000))
mpgv = float(np.clip(INPUT_MPG, 2.8, 235.0))
en   = float(np.clip(INPUT_ENGINESIZE, 0.0, 4.5))

hasil = prediksi_lengkap(yr, mi, tx, mpgv, en)
# Mengecek apakah ada aturan yang aktif (firing strength > 0)
# Perbaikan: Iterasi setiap kondisi dalam aturan
firing_vals = [min(hasil['mu'][k] for k in kondisi) for kondisi, _ in ATURAN]

if all(v == 0 for v in firing_vals):
    st.warning("⚠️ Perhatian: Tidak ada aturan yang aktif dengan input ini.")
    harga_mamdani = 0
    harga_sugeno = 0
    kls_md = "Tidak Ada Aturan"
    kls_sg = "Tidak Ada Aturan"
    selisih = 0
else:
    harga_mamdani = hasil['mamdani']
    harga_sugeno = hasil['sugeno']
    kls_md = hasil['kls_md']
    kls_sg = hasil['kls_sg']
    selisih = hasil['selisih']
persen        = hasil['persen']
mu            = hasil['mu']

badge = {
    'Murah' : '<span class="badge-murah">Murah</span>',
    'Sedang': '<span class="badge-sedang">Sedang</span>',
    'Mahal' : '<span class="badge-mahal">Mahal</span>',
}

# ============================================================
#  HASIL ESTIMASI
# ============================================================

st.subheader('📊 Hasil Estimasi Harga')
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:0.85rem;color:#546E7A;margin-bottom:4px">Metode Mamdani (Centroid)</div>
        <div style="font-size:1.6rem;font-weight:700;color:#0D47A1">Rp {harga_mamdani:,.0f}</div>
        <div style="margin-top:8px">{badge[kls_md]}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="result-card-orange">
        <div style="font-size:0.85rem;color:#546E7A;margin-bottom:4px">Metode Sugeno (Weighted Avg)</div>
        <div style="font-size:1.6rem;font-weight:700;color:#E65100">Rp {harga_sugeno:,.0f}</div>
        <div style="margin-top:8px">{badge[kls_sg]}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background:#F3E5F5;border-radius:12px;padding:1.2rem;border-left:5px solid #6A1B9A;margin-bottom:1rem">
        <div style="font-size:0.85rem;color:#546E7A;margin-bottom:4px">Selisih Kedua Metode</div>
        <div style="font-size:1.6rem;font-weight:700;color:#6A1B9A">Rp {selisih:,.0f}</div>
        <div style="font-size:0.9rem;color:#6A1B9A;margin-top:4px">{persen:.2f}% perbedaan</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================
#  TAB
# ============================================================

tab1, tab2, tab3 = st.tabs(['📈 Kurva Keanggotaan', '🔥 Firing Strength Aturan', '📋 Detail Fuzzifikasi'])

with tab1:
    WARNA = ['#1565C0', '#E65100', '#2E7D32']
    konfig_plot = [
        (U_YEAR, yr, 'year — Tahun', [('Lama','year_Lama'),('Sedang','year_Sedang'),('Baru','year_Baru')], lambda v: str(int(v))),
        (U_MILEAGE, mi, 'mileage — KM', [('Rendah','mil_Rendah'),('Sedang','mil_Sedang'),('Tinggi','mil_Tinggi')], lambda v: f'{int(v/1000)}k'),
        (U_TAX, tx, 'tax — Pajak', [('Murah','tax_Murah'),('Normal','tax_Normal'),('Mahal','tax_Mahal')], lambda v: f'{v/1e6:.0f}jt'),
        (U_MPG, mpgv, 'mpg — BBM', [('Boros','mpg_Boros'),('Irit','mpg_Irit'),('Sangat Irit','mpg_SangatIrit')], lambda v: f'{v:.0f}'),
        (U_ENGINESIZE, en, 'engineSize — Mesin', [('Kecil','eng_Kecil'),('Sedang','eng_Sedang'),('Besar','eng_Besar')], lambda v: f'{v:.1f}'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('#F8F9FA')
    for idx, (univ, val, judul, himpunan, fmt_x) in enumerate(konfig_plot):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor('#FFFFFF')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        for sp in ['top','right']:
            ax.spines[sp].set_visible(False)
        for i, (nama, key) in enumerate(himpunan):
            mf_v = MF[key]
            ax.plot(univ, mf_v, color=WARNA[i], linewidth=2, label=nama, zorder=3)
            ax.fill_between(univ, mf_v, alpha=0.08, color=WARNA[i])
            mu_v = float(np.interp(val, univ, mf_v))
            if mu_v > 0.001:
                ax.plot([val,val],[0,mu_v], color=WARNA[i], linewidth=1.3, linestyle=':', zorder=4)
                ax.plot(val, mu_v, 'o', color=WARNA[i], markersize=7, zorder=6, markeredgecolor='white', markeredgewidth=1.2)
                ax.text(val, mu_v+0.07, f'µ={mu_v:.2f}', ha='center', fontsize=8, fontweight='bold', color=WARNA[i])
        ax.axvline(val, color='#C62828', linewidth=1.8, linestyle='--', zorder=5, alpha=0.8)
        ax.set_xlim(univ[0], univ[-1])
        ticks = np.linspace(univ[0], univ[-1], 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt_x(v) for v in ticks], fontsize=8)
        ax.set_ylim(-0.05, 1.35)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(judul, fontsize=10, fontweight='bold', color='#0D47A1')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax6 = axes[1][2]
    ax6.axis('off')
    ax6.text(0.5, 0.5,
             f"Mamdani\nRp {harga_mamdani:,.0f}\n({kls_md})\n\nSugeno\nRp {harga_sugeno:,.0f}\n({kls_sg})\n\nSelisih: {persen:.2f}%",
             ha='center', va='center', fontsize=11, fontfamily='monospace', color='#0D47A1', transform=ax6.transAxes,
             bbox=dict(boxstyle='round,pad=0.8', fc='#E3F2FD', ec='#1565C0', lw=1.5))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    # Buat nama_rule secara dinamis dari ATURAN
    nama_rule = [f"R{i+1:02d}: {' & '.join(kondisi)} → {output}" for i, (kondisi, output) in enumerate(ATURAN)]
    firing_vals = [min(mu[k] for k in kondisi) for kondisi, _ in ATURAN]
    output_vals = [out for _, out in ATURAN]

    df_rules = pd.DataFrame({
        'Aturan'          : nama_rule,
        'Output'          : output_vals,
        'Firing Strength' : [round(v, 4) for v in firing_vals],
        'Aktif'           : ['✓' if v > 0 else '✗' for v in firing_vals],
    })

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(df_rules.style.background_gradient(subset=['Firing Strength'], cmap='Blues'), use_container_width=True, height=500)
    with col_b:
        aktif = sum(1 for v in firing_vals if v > 0)
        st.metric('Aturan Aktif', f'{aktif} / {len(ATURAN)}')
        st.metric('Max Firing', f'{max(firing_vals):.4f}')
        st.metric('Rule Terkuat', nama_rule[firing_vals.index(max(firing_vals))][:20]+'...')

        fig2, ax2 = plt.subplots(figsize=(5, 6))
        fig2.patch.set_facecolor('#F8F9FA')
        warna_bar = ['#C62828' if o=='Mahal' else '#1565C0' if o=='Murah' else '#E65100' for o in output_vals]
        ax2.barh(range(len(ATURAN)), firing_vals, color=warna_bar, alpha=0.8, height=0.7)
        ax2.set_yticks(range(len(ATURAN)))
        ax2.set_yticklabels([f'R{i+1:02d}' for i in range(len(ATURAN))], fontsize=8)
        ax2.set_xlabel('Firing Strength', fontsize=9)
        ax2.set_title('Kekuatan Aktivasi Tiap Aturan', fontsize=10, fontweight='bold', color='#0D47A1')
        ax2.set_facecolor('#FFFFFF')
        for sp in ['top','right']:
            ax2.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

with tab3:
    data_fuzzy = {
        'Variabel'    : ['year']*3 + ['mileage']*3 + ['tax']*3 + ['mpg']*3 + ['engineSize']*3,
        'Himpunan'    : ['Lama','Sedang','Baru', 'Rendah','Sedang','Tinggi', 'Murah','Normal','Mahal', 'Boros','Irit','Sangat Irit', 'Kecil','Sedang','Besar'],
        'Derajat (µ)' : [round(mu['year_Lama'],4), round(mu['year_Sedang'],4), round(mu['year_Baru'],4),
                         round(mu['mil_Rendah'],4), round(mu['mil_Sedang'],4), round(mu['mil_Tinggi'],4),
                         round(mu['tax_Murah'],4), round(mu['tax_Normal'],4), round(mu['tax_Mahal'],4),
                         round(mu['mpg_Boros'],4), round(mu['mpg_Irit'],4), round(mu['mpg_SangatIrit'],4),
                         round(mu['eng_Kecil'],4), round(mu['eng_Sedang'],4), round(mu['eng_Besar'],4)],
    }
    df_fuzz = pd.DataFrame(data_fuzzy)
    def warnai(val):
        if val > 0.7: return 'background-color:#C8E6C9'
        elif val > 0.3: return 'background-color:#FFF9C4'
        elif val > 0: return 'background-color:#FFCCBC'
        return ''
    st.dataframe(df_fuzz.style.applymap(warnai, subset=['Derajat (µ)']), use_container_width=True)

st.divider()
st.markdown("""
<div style="text-align:center; color:#90A4AE; font-size:0.85rem; padding:1rem 0">
    Kelompok My Little Bolu Ketan · IF-48-06 · Sistem Inferensi Fuzzy Mamdani & Sugeno · From Scratch
</div>
""", unsafe_allow_html=True)