import numpy as np
import torch 
import selfies as sf 
from robot.tasks.molecules.selfies_vae.model_positional_unbounded import SELFIESDataset, InfoTransformerVAE
from robot.tasks.molecules.selfies_vae.data import collate_fn
from robot.latent_space_objective import LatentSpaceObjective
from robot.tasks.molecules.mol_utils.mol_utils import GUACAMOL_TASK_NAMES, smiles_to_desired_scores
from robot.tasks.molecules.mol_utils.mol_utils import get_fps_efficient, get_fp_and_fpNbits_from_smile

# Install Molecules Dependencies: 
# pip install rdkit-pypi
# pip install guacamol
# pip install selfies
# pip install networkx
# apt update
# apt install -y build-essential
# apt install -y libxrender1 libxext6 software-properties-common apt-utils
# conda install -y pomegranate 
# pip install --no-deps molsets
# pip install fcd-torch


class MoleculeObjective(LatentSpaceObjective):
    '''MoleculeObjective class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        task_id='pdop',
        path_to_vae_statedict="../lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        xs_to_scores_dict={},
        max_string_length=1024,
        num_calls=0,
        smiles_to_selfies={},
        lb=None,
        ub=None,
    ):
        assert task_id in GUACAMOL_TASK_NAMES + ["logp"]
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.smiles_to_selfies      = {} # dict to hold computed mappings form smiles to selfies strings
        self.fp_sz_dict             = {} # dict to hold computed fingerprints and sizes for molecules (to avoid computing this more than once per molecule)
        self.smiles_to_selfies     = smiles_to_selfies # dict to hold computed mappings form smiles to selfies strings

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
            dim=256, #  SELFIES VAE DEFAULT LATENT SPACE DIM
            lb=lb,
            ub=ub,
        )


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()
        # sample molecular string form VAE decoder
        sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        # grab decoded selfies strings
        decoded_selfies = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        # decode selfies strings to smiles strings (SMILES is needed format for oracle)
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)
            # save smile to selfie mapping to map back later if needed
            self.smiles_to_selfies[smile] = selfie

        return decoded_smiles


    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # method assumes x is a single smiles string
        score = smiles_to_desired_scores(
                    [x], 
                    self.task_id,
                    ).item()
        
        return score


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)
        # load in state dict of trained model:
        if self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict) 
            self.vae.load_state_dict(state_dict, strict=True) 
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length


    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        X_list = []
        for smile in xs_batch:
            try:
                ''' Since there are multiple ways to represent a
                    smile smiles string as a SELFIE, sometimes 
                    sf.encoder gives a selfies strings with rare
                    tokens that can't be tokenized by the dataobj
                    since they weren't in the initial train set,
                    in this case we rely on the smiles_to_selfies
                    dict where we record all initial decoded selfies
                    that were transformed to smiles during optimization
                    '''
                selfie = sf.encoder(smile)
                tokenized_selfie = self.dataobj.tokenize_selfies([selfie])[0]
                encoded_selfie = self.dataobj.encode(tokenized_selfie).unsqueeze(0)
            except:
                selfie = self.smiles_to_selfies[smile]
                tokenized_selfie = self.dataobj.tokenize_selfies([selfie])[0]
                encoded_selfie = self.dataobj.encode(tokenized_selfie).unsqueeze(0)
            X_list.append(encoded_selfie)
        X = collate_fn(X_list)
        dict = self.vae(X.cuda())
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,self.dim)

        return z, vae_loss
    
    def divf(self, smile1, smile2):
        # Diverssity betweeen two mols is -(fingerprint_similarity)
        try:
            fp1, sz1 = self.fp_sz_dict[smile1]
        except:
            fp1, sz1 = get_fp_and_fpNbits_from_smile(smile1)
            self.fp_sz_dict[smile1] = (fp1, sz1) 

        try:
            fp2, sz2 = self.fp_sz_dict[smile2]
        except:
            fp2, sz2 = get_fp_and_fpNbits_from_smile(smile2)
            self.fp_sz_dict[smile2] = (fp2, sz2) 
        
        # if one of the smiles is invalid, 
        #   return infinite similarity to ensure throw out
        #   the new invalid molecule 
        if fp1 is None or fp2 is None:
            fps = np.inf 
        else:
            fps = get_fps_efficient(fp1, sz1, fp2, sz2)

        return -1 * fps


if __name__ == "__main__":
    # testing molecule objective
    obj1 = MoleculeObjective(task_id='pdop' ) 
    print(obj1.num_calls)
    dict1 = obj1(torch.randn(10,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(3,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(1,256))
    print(dict1['scores'], obj1.num_calls)
