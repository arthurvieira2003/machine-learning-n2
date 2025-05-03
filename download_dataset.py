import os
import urllib.request

def download_file(url, filename):
    """
    Baixa um arquivo da URL especificada para o caminho especificado.
    
    Args:
        url: URL do arquivo a ser baixado
        filename: Caminho onde o arquivo será salvo
    """
    print(f"Baixando arquivo de {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Arquivo salvo com sucesso em {filename}")
    except Exception as e:
        print(f"Erro ao baixar o arquivo: {e}")

if __name__ == "__main__":
    # Criar diretório de dados se não existir
    os.makedirs("data", exist_ok=True)
    
    # URL do dataset de fraudes em cartões de crédito
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    
    # Caminho onde salvar o arquivo
    filename = os.path.join("data", "creditcard.csv")
    
    # Baixar o arquivo
    download_file(url, filename) 