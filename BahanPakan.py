import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.optimize import linprog
import base64
from pathlib import Path
import datetime
import io
from fpdf import FPDF

# Function to load and encode a local image
def get_img_as_base64(file_path):
    img_path = Path(file_path)
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Set page configuration
st.set_page_config(
    page_title="Kalkulator Efisiensi Pakan",
    page_icon="🐔", # Changed from cow to chicken emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Kalkulator Efisiensi Pakan')
st.write("""
Aplikasi ini membantu anda menghitung efisiensi ekonomis bahan pakan berdasarkan kandungan nutrisi dan harganya.
Efisiensi dihitung sebagai jumlah nutrisi yang didapat per rupiah yang dikeluarkan.
""")
# Hide default Streamlit elements
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Modified load_sample_data function that returns a single DataFrame
def load_sample_data():
    data = {
        'Bahan': [
            'Jagung', 'Dedak Padi', 'Bungkil Kedelai', 'Tepung Ikan', 'Bungkil Kelapa', 
            'Tepung Daging', 'Pollard', 'Molases', 'Bungkil Sawit', 'Onggok',
            'Limestone', 'Dicalcium Phosphate', 'Salt', 'Mineral Premix'
        ],
        'Protein (%)': [8.5, 12.0, 45.0, 58.0, 20.0, 50.0, 15.0, 4.0, 16.0, 2.5, 0.0, 0.0, 0.0, 0.0],
        'Lemak (%)': [3.8, 13.0, 1.5, 9.0, 1.8, 8.0, 4.0, 0.1, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        'Serat (%)': [2.5, 13.0, 6.0, 1.0, 15.0, 2.5, 10.0, 0.0, 18.0, 3.5, 0.0, 0.0, 0.0, 0.0],
        'ME (Kcal/kg)': [3350, 2800, 2230, 2900, 2200, 2400, 1900, 2600, 1800, 2900, 0, 0, 0, 0],
        'Kalsium (%)': [0.02, 0.10, 0.30, 4.00, 0.20, 8.00, 0.15, 0.70, 0.25, 0.10, 38.0, 24.0, 0.0, 5.0],
        'Fosfor (%)': [0.28, 1.50, 0.65, 2.80, 0.60, 4.00, 0.90, 0.08, 0.55, 0.10, 0.02, 18.5, 0.0, 1.8],
        'Sodium (%)': [0.02, 0.05, 0.03, 0.80, 0.05, 0.70, 0.03, 0.20, 0.04, 0.02, 0.0, 0.0, 39.0, 1.5],
        'Magnesium (%)': [0.12, 0.95, 0.28, 0.16, 0.30, 0.10, 0.50, 0.30, 0.25, 0.05, 0.5, 0.5, 0.0, 1.0],
        'Besi (mg/kg)': [45, 300, 120, 150, 500, 80, 100, 60, 380, 30, 200, 600, 20, 4000],
        'Zinc (mg/kg)': [20, 80, 40, 90, 50, 100, 90, 5, 30, 15, 10, 15, 0, 5000],
        'Copper (mg/kg)': [3, 20, 15, 8, 30, 10, 12, 8, 25, 5, 0, 0, 0, 1000],
        'Mangan (mg/kg)': [6, 120, 30, 5, 60, 12, 70, 30, 40, 15, 40, 20, 0, 1500],
        'Selenium (mg/kg)': [0.1, 0.2, 0.3, 1.8, 0.2, 0.5, 0.2, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0, 30.0],
        'Harga (Rp/kg)': [5000, 3500, 9000, 15000, 4000, 12000, 4500, 3000, 3200, 2000, 1500, 8000, 3000, 25000]
    }
    
    # Return just one DataFrame with all the data
    return pd.DataFrame(data)

# Tampilkan contoh data
with st.expander("Lihat contoh data"):
    df_contoh = load_sample_data()
    
    nutrient_cols = [col for col in df_contoh.columns if col != 'Bahan' and col != 'Harga (Rp/kg)']
    
    st.subheader("Contoh Data Kandungan dan Harga Bahan Pakan")
    st.dataframe(df_contoh)
    
    st.download_button(
        label="Download contoh data (CSV)",
        data=df_contoh.to_csv(index=False),
        file_name="contoh_bahan_pakan.csv",
        mime="text/csv"
    )

# Change the data loading section to handle CSV and Excel files
st.header('Data Bahan Pakan')
uploaded_file = st.file_uploader("Upload file data bahan pakan (CSV atau Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df_combined = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_combined = pd.read_excel(uploaded_file)
            
        st.success("Data berhasil diupload!")
        st.dataframe(df_combined)
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Menggunakan data contoh sebagai pengganti.")
        df_combined = load_sample_data()
else:
    st.info("Tidak ada file yang diupload. Menggunakan data contoh.")
    df_combined = load_sample_data()

# Display all columns
if st.checkbox("Tampilkan deskripsi kolom data"):
    st.write("### Deskripsi Kolom Data")
    nutrient_cols = [col for col in df_combined.columns if col != 'Bahan' and col != 'Harga (Rp/kg)']
    
    for col in nutrient_cols:
        st.write(f"- **{col}**: Kandungan {col.split(' ')[0]} dalam bahan pakan")
    
    st.write("- **Harga (Rp/kg)**: Harga bahan pakan per kilogram dalam Rupiah")

# Download template file
st.download_button(
    label="Download template data (CSV)",
    data=load_sample_data().to_csv(index=False),
    file_name="template_bahan_pakan.csv",
    mime="text/csv"
)

# Menghitung efisiensi pakan
st.header('Hasil Perhitungan Efisiensi Pakan')

# Pastikan kolom Bahan ada di kedua DataFrame
if 'Bahan' in df_combined.columns:
    # Gabungkan data kandungan dan harga
    df_merged = df_combined
    
    if len(df_merged) > 0:
        # Hitung efisiensi untuk setiap nutrisi
        nutrient_cols = [col for col in df_combined.columns if col != 'Bahan']
        
        df_efisiensi = pd.DataFrame({'Bahan': df_merged['Bahan']})
        
        for nutrient in nutrient_cols:
            df_efisiensi[f'{nutrient}'] = df_merged[nutrient]
            df_efisiensi[f'Efisiensi {nutrient} per Rp'] = df_merged[nutrient] / df_merged['Harga (Rp/kg)']
        
        df_efisiensi['Harga (Rp/kg)'] = df_merged['Harga (Rp/kg)']
        
        # Tampilkan hasil
        st.subheader("Tabel Efisiensi Bahan Pakan")
        st.dataframe(df_efisiensi)
        
        # Download hasil
        st.download_button(
            label="Download hasil perhitungan efisiensi (CSV)",
            data=df_efisiensi.to_csv(index=False),
            file_name="hasil_efisiensi_pakan.csv",
            mime="text/csv"
        )
        
        # Visualisasi efisiensi untuk setiap nutrisi
        st.subheader("Visualisasi Efisiensi Nutrisi")
        
        nutrient_to_visualize = st.selectbox(
            "Pilih nutrisi untuk visualisasi:",
            [col for col in nutrient_cols]
        )
        
        # 1. Visualisasi Efisiensi (Nutrisi per Rp)
        fig1 = px.bar(
            df_efisiensi.sort_values(f'Efisiensi {nutrient_to_visualize} per Rp', ascending=False),
            x='Bahan',
            y=f'Efisiensi {nutrient_to_visualize} per Rp',
            title=f"Efisiensi {nutrient_to_visualize} per Rupiah",
            color='Bahan',
            text_auto='.3f'
        )
        fig1.update_layout(xaxis_title="Bahan Pakan", yaxis_title=f"{nutrient_to_visualize} per Rupiah")
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Scatter plot kandungan nutrisi vs harga
        fig2 = px.scatter(
            df_efisiensi,
            x='Harga (Rp/kg)',
            y=nutrient_to_visualize,
            text='Bahan',
            size=f'Efisiensi {nutrient_to_visualize} per Rp',
            hover_data=[nutrient_to_visualize, 'Harga (Rp/kg)', f'Efisiensi {nutrient_to_visualize} per Rp'],
            title=f"Perbandingan Kandungan {nutrient_to_visualize} vs Harga"
        )
        fig2.update_traces(textposition='top center')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Interpretasi hasil
        st.subheader("Interpretasi Hasil")
        
        # Temukan bahan pakan teratas berdasarkan efisiensi
        best_protein = df_efisiensi.sort_values('Efisiensi Protein (%) per Rp', ascending=False).iloc[0]
        best_energy = df_efisiensi.sort_values('Efisiensi ME (Kcal/kg) per Rp', ascending=False).iloc[0]
        
        st.markdown(f"""
        ### Analisis Efisiensi
        
        **Protein**: Bahan pakan dengan efisiensi protein tertinggi adalah **{best_protein['Bahan']}** 
        dengan nilai {best_protein['Efisiensi Protein (%) per Rp']:.4f}% protein per rupiah. 
        Ini artinya setiap Rp 1,000 yang dikeluarkan untuk membeli {best_protein['Bahan']} 
        akan memberikan {best_protein['Efisiensi Protein (%) per Rp']*1000:.2f}% protein.
        
        **Energi**: Bahan pakan dengan efisiensi energi tertinggi adalah **{best_energy['Bahan']}** 
        dengan nilai {best_energy['Efisiensi ME (Kcal/kg) per Rp']:.4f} Kcal per rupiah.
        
        **Panduan Penggunaan Hasil**:
        1. Efisiensi tinggi artinya bahan tersebut memberikan nutrisi lebih banyak per rupiah yang dikeluarkan
        2. Formulasi pakan dapat mengoptimalkan penggunaan bahan dengan efisiensi tinggi
        3. Perhatikan batasan penggunaan setiap bahan (misalnya: batas maksimal penggunaan dedak dalam ransum)
        4. Pastikan ransum akhir tetap memenuhi kebutuhan nutrisi hewan ternak
        """)

        # Bagian optimasi ransum
        st.header('Optimasi Formulasi Ransum')
        st.write("""
        Formulasikan ransum dengan kombinasi bahan pakan yang optimal untuk memenuhi kebutuhan 
        nutrisi (termasuk mineral) dengan biaya minimum.
        """)

        # Tab untuk berbagai kategori nutrisi
        tab1, tab2, tab3 = st.tabs(["Nutrisi Utama", "Mineral Makro", "Mineral Mikro"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                target_protein = st.number_input('Target Protein (%)', min_value=0.0, max_value=50.0, value=18.0, step=0.5)
            with col2:
                target_energy = st.number_input('Target Energi (Kcal/kg)', min_value=1000, max_value=4000, value=2800, step=50)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                target_calcium = st.number_input('Target Kalsium (%)', min_value=0.0, max_value=5.0, value=0.9, step=0.1)
                target_phosphorus = st.number_input('Target Fosfor (%)', min_value=0.0, max_value=2.0, value=0.6, step=0.1)
            with col2:
                target_sodium = st.number_input('Target Sodium (%)', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
                target_magnesium = st.number_input('Target Magnesium (%)', min_value=0.0, max_value=0.5, value=0.2, step=0.05)
                
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                target_iron = st.number_input('Target Besi (mg/kg)', min_value=0.0, max_value=500.0, value=80.0, step=5.0)
                target_zinc = st.number_input('Target Zinc (mg/kg)', min_value=0.0, max_value=500.0, value=50.0, step=5.0)
                target_copper = st.number_input('Target Copper (mg/kg)', min_value=0.0, max_value=100.0, value=10.0, step=1.0)
            with col2:
                target_manganese = st.number_input('Target Mangan (mg/kg)', min_value=0.0, max_value=300.0, value=60.0, step=5.0)
                target_selenium = st.number_input('Target Selenium (mg/kg)', min_value=0.0, max_value=1.0, value=0.3, step=0.05)

        # Tampilkan penjelasan metode optimasi
        with st.expander("Informasi tentang metode optimasi"):
            st.markdown("""
            ### Metode Optimasi Linear Programming
            
            Aplikasi ini menggunakan teknik Linear Programming untuk mencari kombinasi bahan pakan yang:
            
            1. **Memenuhi kebutuhan nutrisi minimum** (protein dan energi)
            2. **Meminimalkan biaya total ransum**
            3. **Memperhatikan batasan penggunaan** masing-masing bahan
            
            Algoritma ini akan menghitung persentase optimal dari setiap bahan pakan dalam ransum final.
            """)

        # Form untuk batasan penggunaan bahan
        with st.expander("Atur batasan penggunaan bahan (opsional)"):
            st.info("Tentukan batasan minimum dan maksimum persentase untuk setiap bahan dalam ransum (0-100%)")
            
            constraints = {}
            
            for bahan in df_merged['Bahan']:
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input(f"Min {bahan} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"min_{bahan}")
                with col2:
                    max_val = st.number_input(f"Max {bahan} (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"max_{bahan}")
                constraints[bahan] = (min_val/100, max_val/100)  # Convert to proportion (0-1)

        st.subheader("Pengaturan Optimasi Lanjutan")

        use_all_ingredients = st.checkbox("Gunakan semua bahan pakan", value=True)
        min_usage_percentage = st.slider("Persentase minimal penggunaan setiap bahan (%)", 
                                        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                        help="Setiap bahan akan digunakan minimal sebesar persentase ini")

        auto_optimize = st.checkbox("Optimasi otomatis (coba-coba)", value=False, 
                                   help="Aplikasi akan mencoba berbagai kombinasi target untuk mendapatkan hasil terbaik")

        if auto_optimize:
            st.info("Optimasi otomatis akan mencoba berbagai kombinasi target protein dan energi untuk menemukan formulasi dengan biaya terendah namun tetap memenuhi kebutuhan minimal.")
            
            # Parameter untuk optimasi otomatis
            col1, col2 = st.columns(2)
            with col1:
                protein_margin = st.slider("Rentang variasi protein (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
                protein_steps = st.slider("Jumlah langkah protein", min_value=3, max_value=10, value=5)
            with col2:
                energy_margin = st.slider("Rentang variasi energi (Kcal)", min_value=50, max_value=500, value=200, step=50)
                energy_steps = st.slider("Jumlah langkah energi", min_value=3, max_value=10, value=5)

            optimization_metric = st.selectbox(
                "Kriteria optimasi", 
                ["Biaya terendah", "Rasio protein/biaya terbaik", "Rasio energi/biaya terbaik", "Seimbang"],
                index=0
            )

        # Tombol untuk memulai optimasi
        optimize_button = st.button("Optimasi Formulasi Ransum")

        # Modify the run_optimization function to include minerals
        def run_optimization(targets):
            # Extract target values from the targets dictionary
            target_protein_val = targets.get('protein', 0)
            target_energy_val = targets.get('energy', 0)
            
            # Extract mineral targets
            target_calcium_val = targets.get('calcium', 0)
            target_phosphorus_val = targets.get('phosphorus', 0)
            target_sodium_val = targets.get('sodium', 0)
            target_magnesium_val = targets.get('magnesium', 0)
            target_iron_val = targets.get('iron', 0)
            target_zinc_val = targets.get('zinc', 0)
            target_copper_val = targets.get('copper', 0)
            target_manganese_val = targets.get('manganese', 0)
            target_selenium_val = targets.get('selenium', 0)
            
            # Get the various nutrient values from the dataframe
            bahan_list = df_combined['Bahan'].tolist()
            protein_values = df_combined['Protein (%)'].tolist()
            energy_values = df_combined['ME (Kcal/kg)'].tolist()
            calcium_values = df_combined['Kalsium (%)'].tolist()
            phosphorus_values = df_combined['Fosfor (%)'].tolist()
            sodium_values = df_combined['Sodium (%)'].tolist()
            magnesium_values = df_combined['Magnesium (%)'].tolist()
            iron_values = df_combined['Besi (mg/kg)'].tolist()
            zinc_values = df_combined['Zinc (mg/kg)'].tolist()
            copper_values = df_combined['Copper (mg/kg)'].tolist()
            manganese_values = df_combined['Mangan (mg/kg)'].tolist()
            selenium_values = df_combined['Selenium (mg/kg)'].tolist()
            costs = df_combined['Harga (Rp/kg)'].tolist()
            
            n_ingredients = len(bahan_list)
            
            # Objective function: minimize cost
            c = costs
            
            # Inequality constraints (>= target)
            A_ub = []
            b_ub = []
            
            # Add constraints for main nutrients
            if target_protein_val > 0:
                A_ub.append([-val for val in protein_values])
                b_ub.append(-target_protein_val)
                
            if target_energy_val > 0:
                A_ub.append([-val for val in energy_values])
                b_ub.append(-target_energy_val)
            
            # Add constraints for macro minerals
            if target_calcium_val > 0:
                A_ub.append([-val for val in calcium_values])
                b_ub.append(-target_calcium_val)
                
            if target_phosphorus_val > 0:
                A_ub.append([-val for val in phosphorus_values])
                b_ub.append(-target_phosphorus_val)
                
            if target_sodium_val > 0:
                A_ub.append([-val for val in sodium_values])
                b_ub.append(-target_sodium_val)
                
            if target_magnesium_val > 0:
                A_ub.append([-val for val in magnesium_values])
                b_ub.append(-target_magnesium_val)
            
            # Add constraints for micro minerals
            if target_iron_val > 0:
                A_ub.append([-val for val in iron_values])
                b_ub.append(-target_iron_val)
                
            if target_zinc_val > 0:
                A_ub.append([-val for val in zinc_values])
                b_ub.append(-target_zinc_val)
                
            if target_copper_val > 0:
                A_ub.append([-val for val in copper_values])
                b_ub.append(-target_copper_val)
                
            if target_manganese_val > 0:
                A_ub.append([-val for val in manganese_values])
                b_ub.append(-target_manganese_val)
                
            if target_selenium_val > 0:
                A_ub.append([-val for val in selenium_values])
                b_ub.append(-target_selenium_val)
            
            # Add constraints for ingredient usage limits
            for i, bahan in enumerate(bahan_list):
                min_val, max_val = constraints.get(bahan, (0, 1))
                
                # Minimum constraint
                min_constraint = [0] * n_ingredients
                min_constraint[i] = -1
                A_ub.append(min_constraint)
                b_ub.append(-min_val)
                
                # Maximum constraint
                max_constraint = [0] * n_ingredients
                max_constraint[i] = 1
                A_ub.append(max_constraint)
                b_ub.append(max_val)
            
            # Equality constraint: sum of proportions = 1 (100%)
            A_eq = [np.ones(n_ingredients)]
            b_eq = [1.0]
            
            # Bounds for each variable (0 to 1)
            bounds = [(0, 1) for _ in range(n_ingredients)]
            
            # Solve linear programming problem
            try:
                result = linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method='highs'
                )
                
                return result
            except Exception as e:
                print(f"Optimization error: {e}")
                return None

        # When running optimization, combine all targets
        if optimize_button:
            with st.spinner("Memproses optimasi..."):
                # Create targets dictionary
                targets = {
                    'protein': target_protein,
                    'energy': target_energy,
                    'calcium': target_calcium,
                    'phosphorus': target_phosphorus,
                    'sodium': target_sodium,
                    'magnesium': target_magnesium,
                    'iron': target_iron,
                    'zinc': target_zinc,
                    'copper': target_copper,
                    'manganese': target_manganese,
                    'selenium': target_selenium
                }
                
                # Run optimization with all targets
                best_result = run_optimization(targets)
                
                # Siapkan data untuk optimasi
                bahan_list = df_merged['Bahan'].tolist()
                protein_values = df_merged['Protein (%)'].tolist()
                energy_values = df_merged['ME (Kcal/kg)'].tolist()
                costs = df_merged['Harga (Rp/kg)'].tolist()
                
                n_ingredients = len(bahan_list)
                
                # Jika menggunakan semua bahan, update constraints untuk memastikan penggunaan minimal
                if use_all_ingredients:
                    for bahan in df_merged['Bahan']:
                        min_val, max_val = constraints[bahan]
                        constraints[bahan] = (max(min_val, min_usage_percentage/100), max_val)
                        
                # Fungsi untuk melakukan optimasi dengan target tertentu
                def run_optimization(target_protein_val, target_energy_val):
                    # Objective function: minimize cost (c vector)
                    c = costs
                    
                    # Inequality constraints (A_ub and b_ub)
                    # Format: protein >= target_protein, energy >= target_energy
                    A_ub = [
                        [-val for val in protein_values],    # -1 * protein untuk mengubah ke >= constraint
                        [-val for val in energy_values]      # -1 * energy untuk mengubah ke >= constraint
                    ]
                    
                    b_ub = [
                        -target_protein_val,    # -1 * target untuk mengubah ke >= constraint
                        -target_energy_val      # -1 * target untuk mengubah ke >= constraint
                    ]
                    
                    # Tambahkan batasan min dan max untuk setiap bahan
                    for i, bahan in enumerate(bahan_list):
                        min_val, max_val = constraints[bahan]
                        
                        # Batasan minimum (x_i >= min_val) menjadi (-x_i <= -min_val)
                        min_constraint = [0] * n_ingredients
                        min_constraint[i] = -1
                        A_ub.append(min_constraint)
                        b_ub.append(-min_val)
                        
                        # Batasan maximum (x_i <= max_val)
                        max_constraint = [0] * n_ingredients
                        max_constraint[i] = 1
                        A_ub.append(max_constraint)
                        b_ub.append(max_val)
                    
                    # Equality constraint: sum of proportions = 1 (100%)
                    A_eq = [np.ones(n_ingredients)]
                    b_eq = [1.0]
                    
                    # Bounds untuk variabel
                    bounds = [(0, 1) for _ in range(n_ingredients)]
                    
                    # Solve linear programming problem
                    try:
                        result = linprog(
                            c=c,
                            A_ub=A_ub,
                            b_ub=b_ub,
                            A_eq=A_eq,
                            b_eq=b_eq,
                            bounds=bounds,
                            method='highs'
                        )
                        
                        return result
                    except Exception as e:
                        return None
                
                # Jika optimasi otomatis dipilih, jalankan berbagai kombinasi target
                results = []
                best_result = None
                
                if auto_optimize:
                    protein_range = np.linspace(target_protein - protein_margin, target_protein + protein_margin, protein_steps)
                    energy_range = np.linspace(target_energy - energy_margin, target_energy + energy_margin, energy_steps)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_combinations = len(protein_range) * len(energy_range)
                    counter = 0
                    
                    for p_val in protein_range:
                        for e_val in energy_range:
                            counter += 1
                            progress = int(counter / total_combinations * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Mencoba kombinasi {counter}/{total_combinations}: Protein {p_val:.1f}%, Energi {e_val:.0f} Kcal/kg")
                            
                            result = run_optimization(p_val, e_val)
                            
                            if result and result.success:
                                # Hitung total nutrisi dan biaya
                                total_protein = sum(protein_values[i] * result.x[i] for i in range(n_ingredients))
                                total_energy = sum(energy_values[i] * result.x[i] for i in range(n_ingredients))
                                total_cost = sum(costs[i] * result.x[i] for i in range(n_ingredients))
                                
                                # Hanya pertimbangkan jika memenuhi kebutuhan minimum
                                if total_protein >= target_protein and total_energy >= target_energy:
                                    # Menghitung berbagai skor untuk komparasi
                                    protein_cost_ratio = total_protein / total_cost
                                    energy_cost_ratio = total_energy / total_cost
                                    balance_score = (protein_cost_ratio + energy_cost_ratio) / 2
                                    
                                    results.append({
                                        'protein': total_protein,
                                        'energy': total_energy,
                                        'cost': total_cost * 100,  # per 100kg
                                        'protein_cost_ratio': protein_cost_ratio,
                                        'energy_cost_ratio': energy_cost_ratio,
                                        'balance_score': balance_score,
                                        'result': result,
                                        'target_protein': p_val,
                                        'target_energy': e_val
                                    })
                    
                    # Pilih hasil terbaik berdasarkan kriteria
                    if results:
                        if optimization_metric == "Biaya terendah":
                            results.sort(key=lambda x: x['cost'])
                            best_result = results[0]['result']
                            target_protein = results[0]['target_protein']
                            target_energy = results[0]['target_energy']
                        elif optimization_metric == "Rasio protein/biaya terbaik":
                            results.sort(key=lambda x: -x['protein_cost_ratio'])
                            best_result = results[0]['result']
                            target_protein = results[0]['target_protein']
                            target_energy = results[0]['target_energy']
                        elif optimization_metric == "Rasio energi/biaya terbaik":
                            results.sort(key=lambda x: -x['energy_cost_ratio'])
                            best_result = results[0]['result']
                            target_protein = results[0]['target_protein']
                            target_energy = results[0]['target_energy']
                        else:  # Seimbang
                            results.sort(key=lambda x: -x['balance_score'])
                            best_result = results[0]['result']
                            target_protein = results[0]['target_protein']
                            target_energy = results[0]['target_energy']
                        
                        st.success(f"Optimasi otomatis selesai! Ditemukan formulasi optimal dengan target Protein {target_protein:.1f}% dan Energi {target_energy:.0f} Kcal/kg")
                    else:
                        st.warning("Tidak ditemukan formulasi yang memenuhi syarat dalam rentang target yang dicoba.")
                        st.info("Coba perluas rentang variasi atau kurangi persyaratan minimum.")
                else:
                    best_result = run_optimization(target_protein, target_energy)
                
                # Lanjutkan dengan visualisasi hasil
                if best_result and best_result.success:
                    # Buat dataframe hasil optimasi
                    formulation = pd.DataFrame({
                        'Bahan': bahan_list,
                        'Persentase (%)': [round(val * 100, 2) for val in best_result.x],
                        'Jumlah (kg per 100kg)': [round(val * 100, 2) for val in best_result.x],
                        'Protein Kontribusi (%)': [round(p * x * 100, 2) for p, x in zip(df_merged['Protein (%)']/100, best_result.x)],
                        'Energi Kontribusi (Kcal)': [round(e * x, 2) for e, x in zip(df_merged['ME (Kcal/kg)'], best_result.x)],
                        'Biaya Kontribusi (Rp)': [round(c * x * 100, 2) for c, x in zip(df_merged['Harga (Rp/kg)'], best_result.x)]
                    })
                    
                    # Tampilkan hasil optimasi
                    st.subheader("Hasil Optimasi Formulasi Ransum")
                    
                    # Sort by percentage and keep all ingredients (including small percentages)
                    formulation = formulation.sort_values(by='Persentase (%)', ascending=False).reset_index(drop=True)
                    
                    # Highlight bahan dengan persentase kecil
                    def highlight_small_percentage(val):
                        if isinstance(val, float) and val < min_usage_percentage:
                            return 'background-color: yellow'
                        return ''
                        
                    # Tampilkan dataframe dengan highlight
                    st.dataframe(formulation.style.applymap(highlight_small_percentage, subset=['Persentase (%)']))
                    
                    # Hitung total nutrisi dan biaya
                    total_protein = sum(formulation['Protein Kontribusi (%)'])
                    total_energy = sum(formulation['Energi Kontribusi (Kcal)'])
                    total_cost = sum(formulation['Biaya Kontribusi (Rp)'])
                    cost_per_kg = total_cost / 100
                    
                    # Tampilkan ringkasan
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Protein (%)", f"{total_protein:.2f}%", f"{total_protein - target_protein:.2f}%" if total_protein > target_protein else "")
                    with col2:
                        st.metric("Total Energi (Kcal/kg)", f"{total_energy:.0f}", f"{total_energy - target_energy:.0f}" if total_energy > target_energy else "")
                    with col3:
                        st.metric("Biaya per kg (Rp)", f"{cost_per_kg:,.0f}")
                    
                    # Visualisasi komposisi ransum
                    fig = px.pie(
                        formulation, 
                        values='Persentase (%)', 
                        names='Bahan',
                        title='Komposisi Ransum Optimal',
                        hover_data=['Jumlah (kg per 100kg)', 'Biaya Kontribusi (Rp)']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualisasi kontribusi nutrisi
                    fig2 = px.bar(
                        formulation,
                        x='Bahan',
                        y='Protein Kontribusi (%)',
                        title='Kontribusi Protein per Bahan',
                        color='Bahan'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    fig3 = px.bar(
                        formulation,
                        x='Bahan',
                        y='Energi Kontribusi (Kcal)',
                        title='Kontribusi Energi per Bahan',
                        color='Bahan'
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Visualisasi biaya kontribusi
                    fig4 = px.bar(
                        formulation,
                        x='Bahan',
                        y='Biaya Kontribusi (Rp)',
                        title='Kontribusi Biaya per Bahan',
                        color='Bahan'
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    # Download hasil optimasi
                    csv = formulation.to_csv(index=False)
                    st.download_button(
                        label="Download Formulasi Ransum (CSV)",
                        data=csv,
                        file_name="formulasi_ransum_optimal.csv",
                        mime="text/csv"
                    )
                    
                    st.success("Optimasi berhasil! Formulasi ransum di atas memenuhi kebutuhan nutrisi dengan biaya optimal.")
                    
                    # Tambahkan penjelasan hasil optimasi dengan opsi yang dipilih
                    if auto_optimize:
                        st.markdown(f"""
                        ### Interpretasi Hasil Optimasi Otomatis
                        
                        Setelah mencoba {len(protein_range) * len(energy_range)} kombinasi target protein dan energi yang berbeda,
                        formulasi optimal ditemukan dengan target **protein {target_protein:.1f}%** dan **energi {target_energy:.0f} Kcal/kg**.
                        
                        Ransum yang dihasilkan mengandung **{total_protein:.2f}% protein** dan **{total_energy:.0f} Kcal/kg energi**
                        dengan biaya **Rp {cost_per_kg:,.0f} per kg**.
                        
                        **Kriteria optimasi yang digunakan:** {optimization_metric}
                        """)
                    else:
                        st.markdown(f"""
                        ### Interpretasi Hasil Optimasi
                        
                        Ransum optimal yang dihasilkan mengandung {total_protein:.2f}% protein dan {total_energy:.0f} Kcal/kg energi 
                        dengan biaya Rp {cost_per_kg:,.0f} per kg. Formulasi ini merupakan kombinasi optimal untuk mencapai
                        target protein {target_protein}% dan energi {target_energy} Kcal/kg.
                        
                        {"Semua bahan pakan digunakan dengan persentase minimal " + str(min_usage_percentage) + "%" if use_all_ingredients else "Beberapa bahan pakan mungkin tidak digunakan jika tidak efisien secara ekonomis."}
                        """)
                        
                    st.markdown(f"""
                    **Rekomendasi Penggunaan:**
                    
                    1. Untuk membuat 100 kg ransum, gunakan bahan-bahan sesuai kolom "Jumlah (kg)"
                    2. Total biaya untuk 100 kg ransum adalah Rp {total_cost:,.0f}
                    3. Formulasi ini sudah optimal untuk target nutrisi yang ditetapkan
                    
                    **Catatan Penting:**
                    - Hasil optimasi ini fokus pada protein dan energi. Nutrisi mikro lainnya perlu diperhatikan
                    - Pertimbangkan menambahkan premix vitamin dan mineral sesuai kebutuhan
                    - Ketersediaan dan kualitas bahan pakan dapat mempengaruhi hasil akhir
                    """)
                    
                    # Tambahkan grafik perbandingan hasil optimasi jika menggunakan auto-optimize
                    if auto_optimize and len(results) > 1:
                        st.subheader("Perbandingan Hasil Optimasi")
                        
                        # Buat dataframe untuk perbandingan
                        comparison_df = pd.DataFrame({
                            'Target Protein (%)': [r['target_protein'] for r in results],
                            'Target Energi (Kcal/kg)': [r['target_energy'] for r in results],
                            'Protein Aktual (%)': [r['protein'] for r in results],
                            'Energi Aktual (Kcal/kg)': [r['energy'] for r in results],
                            'Biaya per 100kg (Rp)': [r['cost'] for r in results],
                            'Rasio Protein/Biaya': [r['protein_cost_ratio'] for r in results],
                            'Rasio Energi/Biaya': [r['energy_cost_ratio'] for r in results],
                        })
                        
                        # Scatter plot biaya vs nutrisi
                        fig5 = px.scatter(
                            comparison_df,
                            x='Biaya per 100kg (Rp)',
                            y='Protein Aktual (%)',
                            size='Energi Aktual (Kcal/kg)',
                            hover_data=['Target Protein (%)', 'Target Energi (Kcal/kg)'],
                            title='Perbandingan Biaya vs Nutrisi dari Berbagai Formulasi',
                            labels={'Biaya per 100kg (Rp)': 'Biaya (Rp per 100kg)', 'Protein Aktual (%)': 'Protein (%)'}
                        )
                        st.plotly_chart(fig5, use_container_width=True)
                        
                        st.dataframe(comparison_df.sort_values('Biaya per 100kg (Rp)').head(10))
                        
                    # Add this after the existing CSV download button in the optimization results section
                    if best_result and best_result.success:
                        # Create Excel file in memory
                        output = io.BytesIO()
                        
                        # Create a Pandas Excel writer using the BytesIO object
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            # Write the formulation data to sheet 1
                            formulation.to_excel(writer, sheet_name='Formulasi Ransum', index=False)
                            
                            # Access the workbook and worksheet objects
                            workbook = writer.book
                            worksheet = writer.sheets['Formulasi Ransum']
                            
                            # Add a summary sheet
                            summary_data = {
                                'Parameter': ['Tanggal', 'Total Protein (%)', 'Target Protein (%)', 
                                             'Total Energi (Kcal/kg)', 'Target Energi (Kcal/kg)',
                                             'Biaya per kg (Rp)', 'Biaya per 100kg (Rp)'],
                                'Nilai': [datetime.datetime.now().strftime("%d-%m-%Y"), 
                                         f"{total_protein:.2f}", f"{target_protein:.2f}",
                                         f"{total_energy:.0f}", f"{target_energy:.0f}",
                                         f"{cost_per_kg:,.0f}", f"{total_cost:,.0f}"]
                            }
                            
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Ringkasan', index=False)
                            
                            # Format the Excel file
                            header_format = workbook.add_format({
                                'bold': True,
                                'text_wrap': True,
                                'valign': 'top',
                                'fg_color': '#D7E4BC',
                                'border': 1
                            })
                            
                            # Set column widths
                            worksheet.set_column('A:A', 20)
                            worksheet.set_column('B:D', 15)
                            worksheet.set_column('E:F', 18)
                        
                        # Reset pointer to beginning of file
                        output.seek(0)
                        
                        # Provide the Excel download button
                        st.download_button(
                            label="Download Formulasi Ransum (Excel)",
                            data=output,
                            file_name="formulasi_ransum_optimal.xlsx",
                            mime="application/vnd.ms-excel",
                            key="download-excel"
                        )
                else:
                    st.error(f"Optimasi tidak berhasil: {best_result.message if best_result else 'Tidak ada solusi yang ditemukan'}")
                    st.info("Coba ubah target nutrisi atau batasan penggunaan bahan")
    else:
        st.warning("Tidak ada kecocokan antara nama bahan di data kandungan dan data harga")
else:
    st.error("Format data tidak sesuai. Pastikan terdapat kolom 'Bahan' pada kedua file CSV")

# Footer with LinkedIn profile link and improved styling
st.markdown("""
<hr style="height:1px;border:none;color:#333;background-color:#333;margin-top:30px;margin-bottom:20px">
""", unsafe_allow_html=True)

# Get current year for footer
current_year = datetime.datetime.now().year

st.markdown(f"""
<div style="text-align:center; padding:15px; margin-top:10px; margin-bottom:20px">
    <p style="font-size:16px; color:#555">
        © {current_year} Developed by: 
        <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank" 
           style="text-decoration:none; color:#0077B5; font-weight:bold">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" 
                 width="16" height="16" style="vertical-align:middle; margin-right:5px">
            Galuh Adi Insani
        </a> 
        with <span style="color:#e25555">❤️</span>
    </p>
    <p style="font-size:12px; color:#777">All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

# Function to create PDF from formulation results
def create_formulation_pdf(formulation, total_protein, target_protein, total_energy, target_energy, total_cost, cost_per_kg):
    pdf = FPDF()
    pdf.add_page()
    
    # Set up the PDF
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Formulasi Ransum Optimal', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Tanggal: {datetime.datetime.now().strftime("%d-%m-%Y")}', 0, 1, 'C')
    pdf.ln(5)
    
    # Add summary metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Ringkasan Nutrisi dan Biaya:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(60, 10, f'Total Protein: {total_protein:.2f}%', 0, 0)
    pdf.cell(60, 10, f'Target Protein: {target_protein:.2f}%', 0, 1)
    pdf.cell(60, 10, f'Total Energi: {total_energy:.0f} Kcal/kg', 0, 0)
    pdf.cell(60, 10, f'Target Energi: {target_energy:.0f} Kcal/kg', 0, 1)
    pdf.cell(60, 10, f'Biaya per kg: Rp {cost_per_kg:,.0f}', 0, 0)
    pdf.cell(60, 10, f'Biaya per 100kg: Rp {total_cost:,.0f}', 0, 1)
    pdf.ln(5)
    
    # Add table header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 10, 'Bahan', 1, 0, 'C')
    pdf.cell(30, 10, 'Persentase (%)', 1, 0, 'C')
    pdf.cell(40, 10, 'Jumlah (kg/100kg)', 1, 0, 'C')
    pdf.cell(60, 10, 'Biaya (Rp)', 1, 1, 'C')
    
    # Add table rows
    pdf.set_font('Arial', '', 10)
    for _, row in formulation.iterrows():
        pdf.cell(60, 10, row['Bahan'], 1, 0)
        pdf.cell(30, 10, f"{row['Persentase (%)']:.2f}", 1, 0, 'R')
        pdf.cell(40, 10, f"{row['Jumlah (kg per 100kg)']:.2f}", 1, 0, 'R')
        pdf.cell(60, 10, f"{row['Biaya Kontribusi (Rp)']:,.2f}", 1, 1, 'R')
    
    # Add footer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, 'Dibuat dengan Kalkulator Efisiensi Pakan by Galuh Adi Insani', 0, 1, 'C')
    
    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')