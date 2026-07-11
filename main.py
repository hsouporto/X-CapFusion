import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="X-CapFusion Pipeline Runner")
    parser.add_argument("--model_type", choices=["earlyLabel", "noLabel", "intermediateLabel", "lateLabel", "textClassifier", "imageClassifier"], required=True,
                        help="Versão do modelo a ser usada")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], required=True,
                        help="Modo de operação")
    parser.add_argument("--config", default=None, help="Caminho opcional para arquivo de configuração YAML")
    parser.add_argument("--dataset", choices=["train", "test"], help="Especifica o dataset para modo 'predict'")

    args = parser.parse_args()

    # Validação condicional
    if args.mode == "predict" and args.dataset is None:
        parser.error("--dataset é obrigatório quando --mode é 'predict'")

    # Importa dinamicamente o módulo correto (ex: earlyLabel.predict)
    try:
        module = importlib.import_module(f"{args.model_type}.{args.mode}")
    except ModuleNotFoundError:
        raise RuntimeError(f"Não foi possível encontrar {args.model_type}.{args.mode}")

    # Executa a função correta
    if hasattr(module, args.mode):
        kwargs = {}
        if args.config:
            kwargs["config_path"] = args.config
        if args.mode == "predict":
            kwargs["dataset"] = args.dataset
        if args.mode == "evaluate":
            kwargs["dataset"] = args.dataset

        getattr(module, args.mode)(**kwargs)
    else:
        raise RuntimeError(f"A função '{args.mode}()' não foi encontrada no módulo {args.model_type}.{args.mode}")

if __name__ == "__main__":
    main()



#python main.py --model_type lateLabel --mode train

#python main.py --model_type intermediateLabel --mode evaluate --dataset test

#python main.py --model_type intermediateLabel --mode predict --dataset test
#python main.py --model_type imageClassifier --mode predict --dataset train
