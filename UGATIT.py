import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from PIL import Image
# from glob import glob

class UGATIT(object) :
    def __init__(self):
        self.light = False

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = 'results'
        self.dataset = 'YOUR_DATASET_NAME'

        self.iteration = 1000000
        self.decay_flag = True

        self.batch_size = 1
        self.print_freq = 1000
        self.save_freq = 100000

        self.lr = 0.0001
        self.weight_decay = 0.0001
        self.ch = 64

        """ Weight """
        self.adv_weight = 1
        self.cycle_weight = 10
        self.identity_weight = 10
        self.cam_weight = 1000

        """ Generator """
        self.n_res = 4

        """ Discriminator """
        self.n_dis = 6

        self.img_size = 256
        self.img_ch = 3

        self.device = 'cpu'
        self.benchmark_flag = False
        self.resume = False

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True



    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        # self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        # self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        # self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        # self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        # self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def load(self,dir):
        params = torch.load(dir, map_location = self.device)
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def run(self):
        real_A, _ = self.testA_loader
        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A[0])
        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

        image = tensor2numpy(denorm(fake_A2B[0]))
        image = np.array(Image.fromarray((image*255).astype(np.uint8)).resize((256,256)).convert('RGB'))
        image = Image.fromarray(image)
        return image