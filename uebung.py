import math
import torch
import gpytorch
from matplotlib import pyplot as plt
#cat klebt 2 Arrays zusammen
# linspace: anfang, ende, anzahl der werte
train_x = torch.cat((torch.linspace(0, 0.6, 50), torch.linspace(0.8, 1, 50)))
# True function is sin(2*pi*x) mit Rauschen
train_y = torch.sin(train_x * (3 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.07)

# Define GP Model
class ExactGPModel(gpytorch.models.ExactGP): 
    # Konstruktor 
    def __init__(self, train_inputs, train_targets, likelihood):
        # Aufruf des Konstruktors der Elternklasse. Man braucht init weil ExactGP von gpytorch.models.ExactGP erbt
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        #kernel 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # falls keine daten dann einfach konstanter mittelwert 
        self.mean_module = gpytorch.means.ConstantMean()


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
         #gibt eine mehrdimensionale normale verteilung zurück Weil wir mehrere Eingabepunkte haben.
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood() # wie stark ist das rauschen in den daten
model = ExactGPModel(train_x, train_y, likelihood) # welches model wir benutzen wollen


training_iter = 150

# Find optimal model hyperparameters
# jetzt bitte einmal lernen
model.train()
likelihood.train()

#optimizer 
#lr steht für learning rate 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters. Passt die parameter des models an damit die vorhersage besser wird
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # Guckt wie schlecht das model ist 

for i in range(training_iter): 
    # Zero gradients from previous iteration also alles löschen was vorher da war und falsch berechnet wurde
    optimizer.zero_grad()
    # Output from model
    output = model(train_x) # Vorhersage des Modells
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)   # Wie schlecht ist die vorhersage also vergleich mit den echten werten
    loss.backward() # was muss ich anpassen damit die vorhersage besser wird
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step() # mach einen schritt in die richtung die der gradient vorgibt
model.covar_module.outputscale = 5              # Maximale Höhe der Blase
# Set into eval mode (fixiert die parameter damit sie sich nicht mehr ändern)
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var(): # keine gradients berechnen weil wir nichts lernen wollen sondern nur vorhersagen
    # Test points are regularly spaced along [0,1]
    test_x = torch.linspace(0, 1, 100)

    # Make predictions by feeding model through likelihood
    observed_pred = likelihood(model(test_x)) # vorhersage des models mit unsicherheiten
    # Verschiedene möglichkeiten die vorhersage zu bekommen
    lower, upper = observed_pred.confidence_region() # Berechnet die grenzen des blauen bereichs




    #f_preds = model(train_x)
    #y_preds = likelihood(model(train_x))
    #f_mean  = f_preds.mean
    #f_var   = f_preds.variance
   # f_covar = f_preds.covariance_matrix
    #f_samples = f_preds.sample(sample_shape=torch.Size([1000],)) # 1000 samples drawn from the function posterior also 1000 mögliche funktionen die zu den daten passen

    f ,ax  =    plt.subplots(1,1,figsize=(4,3))
    # Get upper and lower confidence bounds      
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*') # so macht man sterne für die datenpunkte
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b') # so macht man die blaue linie für den mittelwert
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5) # zeichnet den blauen bereich für die unsicherheit
    ax.set_ylim([-3, 3])        
    ax.set_title('Beispiel Projekt GPs')
    ax.legend(['Datenpunkte', 'Durschnitt', 'Wie sicher sich das model ist'])   
    plt.show()