import numpy as np

def calculate_depth_metrics(depth_gt, depth_pred, mask, threshold_outlier=0.05):
    """
    Calcula métricas 2D considerando apenas pixels válidos pela máscara.
    """
    # Extrair apenas os vetores lineares dos pixels válidos
    gt = depth_gt[mask]
    pred = depth_pred[mask]
    
    if len(gt) == 0:
        return {}

    # 2. Calcular o vetor de Erro Absoluto por pixel válido
    erros_absolutos = np.abs(gt - pred)
    
    # 3. Cálculo das Métricas Estatísticas
    mae = np.mean(erros_absolutos)
    mediana = np.median(erros_absolutos)
    rmse = np.sqrt(np.mean(erros_absolutos ** 2))
    
    # Percentis para análise de cauda (Outliers)
    p90 = np.percentile(erros_absolutos, 90)
    p95 = np.percentile(erros_absolutos, 95)
    p99 = np.percentile(erros_absolutos, 99)
    
    # Moda (Agrupando com precisão de 2 casas decimais para relevância prática)
    valores_arredondados = np.round(erros_absolutos, 2)
    valores, contagens = np.unique(valores_arredondados, return_counts=True)
    moda = valores[np.argmax(contagens)]
    
    # 4. Diagnóstico Automático
    alerta_outlier = bool((rmse - mae) > threshold_outlier)
    
    return {
        "MAE": float(mae),
        "Mediana": float(mediana),
        "Moda": float(moda),
        "RMSE": float(rmse),
        "Percentil_90": float(p90),
        "Percentil_95": float(p95),
        "Percentil_99": float(p99),
        "Alerta_Outliers": alerta_outlier
    }