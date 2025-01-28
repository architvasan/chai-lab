import chai_lab.chai1 as chai1
from pathlib import Path
import torch
def test(
        template_pdbs: str,
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
    #print(dir(cont))
    print(f"temp cont: {cont.template_context}")
    print(f"emb cont: {dir(cont.embedding_context)}")
    print(f"struc cont: {cont.structure_context}")
    import sys
    #sys.exit()
    chai1.run_folding_on_context(cont,
                           output_dir
                            )
    return

if __name__=="__main__":
    import argparse

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Which test to run?")
    
    # Add arguments
    parser.add_argument("-t", "--test", type=str, help="Test type: monomer, complex")
    
    # Parse arguments
    args = parser.parse_args()

    if args.test == "monomer":
        test(
            str('input_tests/meta3.pdb'),
            Path('input_tests/nmnat2.fasta'),
            Path('output_tests/templating_nmnat2')
            )
    elif args.test == "complex":

        test(
            str('input_tests/6yz1_A.pdb'),
            Path('input_tests/nsp10_nsp16.fasta'),
            Path('output_tests/templating_nsp10_nsp16')
            )


#test(
#    "None",
#    Path('input_tests/nsp10_nsp16.fasta'),
#    Path('output_tests/no_templating')
#    )