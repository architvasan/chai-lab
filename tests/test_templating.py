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
    #print(f"temp cont: {cont.template_context}")
    for att in dir(cont.embedding_context):
        print(att)
    #print(f"emb cont: {dir(cont.embedding_context)}")
    #print(f"struc cont: {cont.structure_context}")
    import sys
    #sys.exit()
    chai1.run_folding_on_context(cont,
                           output_dir
                            )
    return

def test_subsample(
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
    for attr, value in cont.embedding_context.__dict__.items():
        print (attr, value)
        print(value.size())
    esm_embeds_trunc = cont.embedding_context.esm_embeddings[300:,]
    setattr(cont.embedding_context, "esm_embeddings", esm_embeds_trunc)
    print(cont.embedding_context.esm_embeddings.size())

    for attr, value in cont.structure_context.__dict__.items():
        print (attr)
        print(value)
        #print(value.size())



    #for att in dir(cont.embedding_context):
    #    print(att)
    #    print(cont.embedding_context.att)
    #print(f"temp cont: {cont.template_context}")
    #print(f"emb cont: {dir(cont.embedding_context)}")
    #print(f"struc cont: {cont.structure_context}")
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
    parser.add_argument("-t", "--test", type=str, help="Test type: monomer, complex, subsample")
    
    # Parse arguments
    args = parser.parse_args()

    if args.test == "monomer":
        test(
            str('input_tests/efgr_ep_A.pdb'),
            Path('input_tests/efgr_epitope.fasta'),
            Path('output_tests/templating_efgr_ep')
            )
    elif args.test == "complex":

        test(
            str('input_tests/meta3.pdb'),
            Path('input_tests/nmnat2_fbx.fasta'),
            Path('output_tests/templating_nmnat2_fbx')
            )
    elif args.test == "subsample":
        test_subsample(
            'input_tests/meta3.pdb',
            Path('input_tests/nmnat2_fbx.fasta'),
            Path('output_tests/templating_nmnat2_fbx_sub')
        )

#test(
#    "None",
#    Path('input_tests/nsp10_nsp16.fasta'),
#    Path('output_tests/no_templating')
#    )
