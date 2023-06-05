import argparse
from pathlib import Path
from typing import Callable, Tuple
import dask.array as da
import numpy as np
import time

import matplotlib.pyplot as plt

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousPhase
from dasf.ml.cluster.kmeans import KMeans
from dasf.transforms import ArraysToDataFrame, PersistDaskData
from dasf.pipeline import Pipeline
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler


class MyDataset(Dataset):
    """Classe para carregar dados de um arquivo .npy
    """
    def __init__(self, name: str, data_path: str, chunks: str = "32Mb"):
        """Instancia um objeto da classe MyDataset

        Parameters
        ----------
        name : str
            Nome simbolicamente associado ao dataset
        data_path : str
            Caminho para o arquivo .npy
        chunks: str
            Tamanho dos chunks para o dask.array
        """
        super().__init__(name=name)
        self.data_path = data_path
        self.chunks = chunks
        
    def _lazy_load_cpu(self):
        return da.from_zarr(self.data_path, chunks=self.chunks)
    
    def _load_cpu(self):
        return np.load(self.data_path)
    
    @task_handler
    def load(self):
        ...
        
def create_executor(address: str=None) -> DaskPipelineExecutor:
    """Cria um DASK executor

    Parameters
    ----------
    address : str, optional
        Endereço do Scheduler, by default None

    Returns
    -------
    DaskPipelineExecutor
        Um executor Dask
    """
    if address is not None:
        addr = ":".join(address.split(":")[:2])
        port = str(address.split(":")[-1])
        print(f"Criando executor. Endereço: {addr}, porta: {port}")
        return DaskPipelineExecutor(local=False, use_gpu=False, address=addr, port=port)
    else:
        return DaskPipelineExecutor(local=True, use_gpu=False)
        
def create_pipeline(dataset_path: str, executor: DaskPipelineExecutor, pipeline_save_location: str = None) -> Tuple[Pipeline, Callable]:
    """Cria o pipeline DASF para ser executado

    Parameters
    ----------
    dataset_path : str
        Caminho para o arquivo .npy
    executor : DaskPipelineExecutor
        Executor Dask

    Returns
    -------
    Tuple[Pipeline, Callable]
        Uma tupla, onde o primeiro elemento é o pipeline e o segundo é último operador (kmeans.fit_predict), 
        de onde os resultados serão obtidos.
    """
    print("Criando pipeline....")
    # Declarando os operadores necessários
    dataset = MyDataset(name="F3 dataset", data_path=dataset_path)
    envelope = Envelope()
    phase = InstantaneousPhase()
    arrays2df = ArraysToDataFrame()
    # Persist é super importante! Se não cada partial_fit do k-means vai computar o grafo até o momento!
    # Usando persist, garantimos que a computação até aqui já foi feita e está em memória distribuida.
    persist = PersistDaskData()
    # Cria um objeto k-means com 5 clusters
    kmeans = KMeans(n_clusters=5, max_iter=15, init="random", algorithm="full")
    
    # Compondo o pipeline
    pipeline = Pipeline(
        name="F3 seismic attributes",
        executor=executor
    )
    pipeline.add(dataset)
    pipeline.add(envelope, X=dataset)
    pipeline.add(phase, X=dataset)
    pipeline.add(arrays2df, dataset=dataset, envelope=envelope, phase=phase)
    pipeline.add(persist, X=arrays2df)
    pipeline.add(kmeans.fit_predict, X=persist)
    
    if pipeline_save_location is not None:
    	pipeline.visualize(filename=pipeline_save_location)
    
    # Retorna o pipeline e o operador kmeans, donde os resultados serão obtidos
    return pipeline, kmeans.fit_predict

def run(pipeline: Pipeline, last_node: Callable) -> np.ndarray:
    """Executa o pipeline e retorna o resultado

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline a ser executado
    last_node : Callable
        Último operador do pipeline, de onde os resultados serão obtidos

    Returns
    -------
    np.ndarray
        NumPy array com os resultados
    """
    print("Executando pipeline")
    start = time.time()
    pipeline.run()
    res = pipeline.get_result_from(last_node)
    res = res.compute()
    end = time.time()
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o arquivo .npy")
    parser.add_argument("--address", type=str, default=None, help="Endereço do scheduler. Formato: tcp://<ip>:<port>")
    parser.add_argument("--save-pipeline-fig", type=str, default=None, help="Local para salvar a figura do pipeline")
    parser.add_argument("--save-inline-fig", type=str, default=None, help="Local para salvar a figura da inline 0 do k-means")
    args = parser.parse_args()
   
    # Criamos o executor
    executor = create_executor(args.address)
    # Depois o pipeline
    pipeline, last_node = create_pipeline(args.data, executor, pipeline_save_location=args.save_pipeline_fig)
    # Executamos e pegamos o resultado
    res = run(pipeline, last_node)
    print(f"O resultado é um array com o shape: {res.shape}")
    
    # Podemos fazer o reshape e printar a primeira inline
    if args.save_inline_fig is not None:
        res = res.reshape((401, 701, 255))
        import matplotlib.pyplot as plt
        plt.imsave(args.save_inline_fig, res[0], cmap="viridis")
        print(f"Figura da inline 0 salva em {args.save_inline_fig}")
    
