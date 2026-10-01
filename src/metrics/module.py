from torch import nn
import os, json, numpy as np
from tqdm import tqdm
from pathlib import Path
from .metrics_2d import calculate_depth_metrics
from .utils import decode_depth,save_error_heatmap

class Metrics(nn.Module):
    """Classe que implementa o método Colmap para realizar a calibração extrínseca."""
    def __init__(self, *, metrics_gt_path:str, metrics_predicted_path:str, metrics_output_path:str, metrics_max_dist:float=10, metrics_plot_mae:bool=False, **kwargs):
        super(Metrics, self).__init__()
        self.gt_path = metrics_gt_path
        self.pred_path = metrics_predicted_path
        self.out_path = metrics_output_path
        self.max_dist = metrics_max_dist
        self.plot = metrics_plot_mae

    def forward(self, **kwargs):
        self.process_all_files(self.gt_path, self.pred_path, self.out_path, self.max_dist, generate_heatmap=self.plot)

    def process_all_files(ground_truth_path:str, predicted_path:str, output_path:str, max_dist:float=10., step:int=1, generate_heatmap:bool=False):
        # 1. Definir os caminhos das pastas usando Pathlib
        GT_DIR = Path(ground_truth_path)
        PRED_DIR = Path(predicted_path)
        
        OUTPUT_DIR = Path(output_path)
        HEATMAP_DIR = OUTPUT_DIR / f"heatmaps_dist_{max_dist}"
        
        # Criar as pastas de saída caso não existam
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

        # Dicionário estruturado temporário para armazenar as visadas por fotosfera
        # Estrutura: { fotosfera_id: { view_name: metrics_dict } }
        raw_storage = {}

        # 2. Encontrar todos os arquivos .npy na pasta Ground Truth
        # O método .glob("*.npy") lista todos os arquivos que terminam com .npy
        pred_files = sorted([f for f in PRED_DIR.glob('*') if f.suffix.lower() in {'.npy', '.tiff'}])
        
        if not pred_files:
            print(f"[-] Nenhum arquivo .npy encontrado em: {PRED_DIR}")
            return

        print(f"[+] Foram encontrados {len(pred_files)} arquivos para análise. Iniciando processamento...")
        print("-" * 60)

        for pred_path in tqdm(pred_files, total=len(pred_files), desc="     Processing"):
            # Extrai o nome base do arquivo (ex: "face_00" de "/caminho/face_00.npy")
            filename_base = pred_path.stem

            # Quebra o nome pelo caractere '_' para extrair o ID da fotosfera (segundo elemento)
            parts = filename_base.split('_')
            if len(parts) < 2:
                print(f"[AVISO] Padrão de nome inválido para extrair ID: {filename_base}. Pulando...")
                continue
            photosphere_id = parts[1] # "e891c918-fac0-420b-abe8-a50477e02506"

            gt_path = GT_DIR / f"{filename_base}.png"

            depth_gt = decode_depth(os.fspath(gt_path), step=step)
            depth_pred = decode_depth(os.fspath(pred_path), step=step)
            
            # 2. Criar a máscara de validação
            valid_mask = (depth_gt > 0) & (depth_gt < max_dist)
            
            # 5. Calcular as métricas 2D
            metrics = calculate_depth_metrics(depth_gt, depth_pred, valid_mask)
            
            # Garantir conversão de tipos NumPy para floats nativos do Python
            clean_metrics = {k: float(v) for k, v in metrics.items()}
            
            # Organizar na estrutura temporária agrupada por ID da fotosfera
            if photosphere_id not in raw_storage:
                raw_storage[photosphere_id] = {}
            raw_storage[photosphere_id][filename_base] = clean_metrics

            # Salvar mapas visuais
            if generate_heatmap: save_error_heatmap(depth_gt, depth_pred, valid_mask, HEATMAP_DIR / f"{filename_base}.png", max_error_viz=1.5, max_dist=max_dist)
        print("-" * 60) # ------------------------------------------------------------------

        # 3. Construção do JSON Final com Cálculo Estatístico Hierárquico
        final_json_structure = {
            "dataset_global_summary": {},
            "photospheres": {}
        }

        # Listas globais para calcular a média das médias e o desvio padrão das fotosferas
        metric_keys = ["MAE", "RMSE", "Abs_Rel"] # TODO fazer dinamico
        global_accumulator = {k: [] for k in metric_keys}

        print("[+] Calculando médias e desvios padrões por fotosfera e globais...")

        for photo_id, views_dict in raw_storage.items():
            # Acumuladores locais desta fotosfera específica
            photo_accumulator = {k: [] for k in metric_keys}
            
            for view_name, view_metrics in views_dict.items():
                for k in metric_keys:
                    photo_accumulator[k].append(view_metrics[k])
            
            # Calcular Média e Desvio Padrão para ESTA fotosfera
            photo_summary = {}
            for k in metric_keys:
                photo_summary[k] = {
                    "mean": float(np.mean(photo_accumulator[k])),
                    "std": float(np.std(photo_accumulator[k]))
                }
                # Alimentar o acumulador global com a MÉDIA desta fotosfera
                global_accumulator[k].append(photo_summary[k]["mean"])

            # Montar o nó desta fotosfera no JSON
            final_json_structure["photospheres"][photo_id] = {
                "photosphere_summary": photo_summary,
                "views": views_dict
            }

        # 4. Calcular Média e Desvio Padrão GLOBAL entre todas as fotosferas
        global_summary = {}
        for k in metric_keys:
            if global_accumulator[k]:
                global_summary[k] = {
                    "mean": float(np.mean(global_accumulator[k])),
                    "std": float(np.std(global_accumulator[k]))
                }
            else:
                global_summary[k] = {"mean": 0.0, "std": 0.0}
                
        final_json_structure["dataset_global_summary"] = global_summary

        # 5. Gravar o Arquivo JSON em Disco
        json_output_path = OUTPUT_DIR / f"relatorio_estatistico_dist_{max_dist}.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            # indent=2 deia o JSON formatado com quebras de linha legíveis para humanos
            json.dump(final_json_structure, f, indent=2, ensure_ascii=False)

        print(f"[SUCESSO] Relatório JSON estruturado gerado em: {json_output_path}")
        return os.fspath(json_output_path)