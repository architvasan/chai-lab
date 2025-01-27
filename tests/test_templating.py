import chai_lab.chai1 as chai1
from pathlib import Path
import torch
def test(
        template_pdbs: Path,
        fasta_file: Path,
        output_dir: Path,
        use_esm_embeddings: bool = True,
        use_msa_server: bool = False,
        msa_server_url: str = "https://api.colabfold.com",
        msa_directory: Path = None,
        constraint_path: Path = None,
        esm_device: torch.device = torch.device("cpu"),
    ):
    cont = chai1.make_all_atom_feature_context(
                fasta_file,
                output_dir,
                use_esm_embeddings,
                use_msa_server,
                msa_server_url,
                msa_directory,
                constraint_path,
                template_pdbs,
                esm_device,
            ) 
    return