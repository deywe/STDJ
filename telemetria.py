import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

def analyze_and_log(parquet_file="stdj_secure_telemetry.parquet"):
    """
    Analisa o dataset Parquet, reproduz o log detalhado e gera o Heatmap.
    """
    if not os.path.exists(parquet_file):
        print(f"❌ Error: File {parquet_file} not found.")
        print("Please run the simulation script first to generate the dataset.")
        sys.exit(1)
        
    print(f"📊 Loading telemetry data from: {parquet_file}...")
    
    # 1. Carregar o Dataset
    df = pd.read_parquet(parquet_file)
    total_frames = len(df)
    
    # 2. Reproduzir o Log Detalhado
    print("\n" + "="*30 + " DETAILED METRICS " + "="*30)
    
    stable_frames = df[df['status'] == 'STABLE']
    turbulent_frames = df[df['status'] == 'TURBULENCE']
    
    print(f"Total Frames: {total_frames}")
    print(f"Stable Frames: {len(stable_frames)} ({len(stable_frames)/total_frames:.1%})")
    print(f"Turbulent Frames: {len(turbulent_frames)} ({len(turbulent_frames)/total_frames:.1%})")
    print("-" * 70)
    
    if not stable_frames.empty:
        print(f"Avg Warp Speed (Stable): {stable_frames['warp_velocity_c'].mean()/1e10:.2f}e10 c")
        print(f"Min Fidelity (Stable): {stable_frames['veracity'].min():.6f}")
        
    if not turbulent_frames.empty:
        print(f"Avg Warp Speed (Turbulent): {turbulent_frames['warp_velocity_c'].mean()/1e10:.2f}e10 c")
        print(f"Max Turbulence Dip (Veracity): {turbulent_frames['veracity'].min():.6f}")
    
    print("="*78)
    
    # 3. Processar Dados para o Heatmap (Veracidade vs Tempo)
    num_frames = len(df)
    side = int(np.sqrt(num_frames))
    
    if side * side != num_frames:
        padded_len = (side + 1) * (side + 1)
        veracity_data = np.pad(df['veracity'].values, 
                               (0, padded_len - num_frames), 
                               'constant', 
                               constant_values=np.nan)
        matrix = veracity_data.reshape((side + 1, side + 1))
    else:
        matrix = df['veracity'].values.reshape((side, side))

    # 4. Gerar o Heatmap
    print("\n🎨 Generating heatmap of Veracity vs. Time...")
    
    plt.figure(figsize=(10, 8))
    
    # Estilo 'inferno' para intensidade de fidelidade
    sns.heatmap(matrix, cmap="inferno", annot=False, 
                cbar_kws={'label': 'Veracity (Fidelity)'})
    
    plt.title("STDJ Warp Field Fidelity Pattern (Sub-Planck Regime)")
    plt.xlabel("Warp Field Phase Axis (Arbitrary Units)")
    plt.ylabel("Temporal Propagation Axis (Arbitrary Units)")
    
    output_image = "warp_fidelity_heatmap.png"
    plt.savefig(output_image)
    print(f"💾 Heatmap saved as: {output_image}")
    
    # 5. Exibir a Imagem
    print("🖥️ Opening heatmap image...")
    plt.show()

if __name__ == "__main__":
    analyze_and_log()
