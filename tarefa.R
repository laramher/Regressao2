#Modelo de regressao linear simples:
# Yi <- 5 + 2*Xi + Ei

#Número de observações:
n <- 10

#Considere um vetor de zeros para armazenar os valores de X: 
X <- numeric(10)
X
#Armazenando os valores de X para 10 indivíduos X = c(X1=1, X2 = 2, ..., X10=10) no vetor de zeros criado:
for (i in 1:10) {
  X[i] <- i
}
X

#Gerando 10000 respostas de Yi (10000 observações).
#Estatística do teste para H0: beta_0 = 5 e H1: beta_0 != 5;
#Estatística do teste para H0: beta_1 = 2 e H1: beta_1 != 2;
#Estatística do teste para H0: beta_1 = 1.8 e H1: beta_1 != 1.8;
set.seed(236181)
simulacao <- 10000

#Vamos criar uma matriz de ZEROS para armazenar o resultado. Nas linhas teremos os 10mil Yi 
# e nas colunas, as variáveis X.
resultados.mat <- matrix(0, nrow = simulacao, ncol = n)
resultados.mat
#Simulando os valores de Yi:
for(i in 1:simulacao){
  Ei <- rnorm(n, mean = 0, sd = 1)
  resultados.mat[i,] <- 5 + 2*X + Ei #na linha i colocarei o resultado do Yi correspondente ao Xi e ao Ei fornecidos. Fiz isso 10 mil vezes
}
resultados.mat
# Agora,vou transformar essa matriz num dataframe, para adicionar nomes as colunas:
simulacoes.Yi <- as.data.frame(resultados.mat)
colnames(simulacoes.Yi) <- c("Yobs.X1", "Yobs.X2", "Yobs.X3", "Yobs.X4", "Yobs.X5", "Yobs.X6","Yobs.X7", "Yobs.X8", "Yobs.X9", "Yobs.X10")
simulacoes.Yi

#################################################### ESTIMAÇÃO DOS BETAS POR MÍNIMOS QUADRADOS
#A próxima etapa é estimar os beta_0 e beta_1 para cada linha uma das 10mil linhas da minha simulacao:
#para o beta_1_hat preciso de: Xi, Yi, n, X_barra, Y_barra
#para o beta_0_hat preciso de: Y_barra, X_barra e beta_1_hat
estimadores.betas <- function(dataframe){
  # 1- criando df para armazenar os betas que serão estimados:
  resultado.estimadores <- data.frame(beta0_hat = numeric(nrow(dataframe)), beta1_hat = numeric(nrow(dataframe)))
  # 2- Agora, pega a i-ésima linha do dataframe e vou estimar os betas:
  for (i in 1:nrow(dataframe)){
    Y <- as.numeric(dataframe[i, ]) #pega a linha i e armazena em Y
    # Estimando beta1_hat:
    beta1_hat <- sum((X - mean(X)) * (Y - mean(Y))) / sum((X - mean(X))^2)
    # Estimando beta0_hat:
    beta0_hat <- mean(Y) - beta1_hat * mean(X)
    # Armazenando os resultados dos betas:
    resultado.estimadores$beta0_hat[i] <- beta0_hat
    resultado.estimadores$beta1_hat[i] <- beta1_hat
  }
  
  return(resultado.estimadores)
}

# Vou aplicar a função ao dataframe das simulações:
coeficientes <- estimadores.betas(simulacoes.Yi)
head(coeficientes)

########################INTERVALO DE CONFIANÇA PARA OS BETAS
########BETA0
#beta0_hat é estimador nao viesado para beta0
#beta0_hat ~ N(beta0, sigma2*((1/n)+ ((Xbarra)^2)/Sxx)
#Logo:
# (beta0_hat - beta0)/sqrt(sigma2*((1/n)+ ((Xbarra)^2)/Sxx) ~ N(0,1)
# (beta0_hat - beta0)/sqrt(MSE*((1/n)+ ((Xbarra)^2)/Sxx) ~ t(gl=n-2)
#obs: Standard Error SE: sqrt(MSE*((1/n)+ ((Xbarra)^2)/Sxx)
#IC.beta0 superior: beta0_hat + t(n-2; 5%/2) * SE
#IC.beta0 inferior: beta0_hat - t(n-2; 5%/2) * SE

########BETA1
#beta1_hat é estimador não viesado para beta1
#beta1_hat ~ N(beta1, sigma2/Sxx)
#Logo:
# (beta1_hat - beta1)/sqrt(sigma2/Sxx) ~ N(0,1)
# (beta1_hat - beta1)/sqrt(MSE/Sxx) ~ t(gl=n-2)
#obs: Standard Error SE: sqrt(MSE/Sxx) 
#IC.beta1 superior: beta1_hat + t(n-2; 5%/2) * SE
#IC.beta1 inferior: beta1_hat - t(n-2; 5%/2) * SE

# 1### Precisamos conhecer MSE, que é o estimador de sigma2. Para isso, precisamos do SSE.
# SSE = soma dos valores (obs - estimado)^2 = sum(Yi - Yi)^2 para todo i
# gl Erro = n-2 (pois são 2 estimadores)

Yi.observados <- simulacoes.Yi

Yhat.preditos <- function(dfcoeficientes, X){
  #Criando uma nova matriz vazia, para armazenar os Y_hat
  Y_hat_mat <- matrix(0, nrow = nrow(dfcoeficientes), ncol = length(X)) 
  for (i in 1:nrow(dfcoeficientes)){
    #Para cada i do data.frame coeficientes do modelo, crio um Yhat
    Y_hat_mat[i, ] <- coeficientes$beta0_hat[i] + coeficientes$beta1_hat[i] * X
  }
  # Transformar a matriz Yhat num dataframe para melhor visualização
  Y_hat <- as.data.frame(Y_hat_mat)
  # Renomear os nomes das colunas:
  colnames(Y_hat) <- c("Yhat.X1", "Yhat.X2", "Yhat.X3", "Yhat.X4", "Yhat.X5", "Yhat.X6","Yhat.X7", "Yhat.X8", "Yhat.X9", "Yhat.X10")
  return(Y_hat)
}

# Aplicando a função para calcular os Y estimados (preditos)
Y.hat <- Yhat.preditos(coeficientes, X)

head(Yi.observados)
head(Y.hat)

# 2### Cálculo do SSE e MSE:
# SSE = soma dos valores (obs - estimado)^2 = sum(Yi - Yi)^2 para todo i
# MSE = SSE/n-2 (porque sao 2 estimadores

library(dplyr)
calcular.sse.mse <- function(observados, esperados){
  #Criando uma nova matriz vazia os residuos
  residuos.mat <- matrix(0, nrow = nrow(observados), ncol = ncol(observados))
  #Criando uma nova matriz vazia para os residuos ao quadrado
  residuos.quadrados.mat <- matrix(0, nrow = nrow(observados), ncol = ncol(observados))
  
  # Calculando os residuos:
  for (i in 1:nrow(observados)){
    residuos.mat <- observados - esperados
  }
  # Transformar a matriz num dataframe para melhor visualização
  residuos <- as.data.frame(residuos.mat)
  # Renomear os nomes das colunas:
  colnames(residuos) <- c("residuos.X1", "residuos.X2", "residuos.X3", "residuos.X4", "residuos.X5", "residuos.X6","residuos.X7", "residuos.X8", "residuos.X9", "residuos.X10")
  
  
  # Calculando os residuos ao quadrado:
  residuos.quadrados.mat <- residuos^2
  # Transformar a matriz num dataframe para melhor visualização
  residuos.quadrados <- as.data.frame(residuos.quadrados.mat)
  # Renomear os nomes das colunas:
  colnames(residuos.quadrados) <- c("residuos2.X1", "residuos2.X2", "residuos2.X3", "residuos2.X4", "residuos2.X5", "residuos2.X6","residuos2.X7", "residuos2.X8", "residuos2.X9", "residuos2.X10")
  
  #Criando coluna id:
  residuos <- cbind(ID = 1:nrow(residuos), residuos)
  residuos.quadrados <- cbind(ID = 1:nrow(residuos.quadrados), residuos.quadrados)
  
  #Juntando os dois data.frames:
  df_merged <- left_join(residuos, residuos.quadrados, by = "ID")
  
  #Calculando o SSE:
  sse <- rowSums(residuos.quadrados[,-1])
  # Transformar num dataframe para melhor visualização
  sse <- as.data.frame(sse)
  # Renomear os nomes das colunas:
  colnames(sse) <- c("SSE")
  
  #Calculando o MSE:
  mse <- sse/(length(X)-2)
  colnames(mse) <- c("MSE")
  
  #Criando coluna id:
  sse <- cbind(ID = 1:nrow(sse), sse)
  mse <- cbind(ID = 1:nrow(mse), mse)
  df <- left_join(sse, mse, by = "ID")
  
  return(df)
}

var_hat <- calcular.sse.mse(Yi.observados, Y.hat)
head(var_hat)

somaXi2 <- sum(X^2)
Xbarra <- mean(X)
Sxx <- somaXi2 - n*(Xbarra)^2


# 3### Calcular o Standard Error (SE) para beta0 em cada simulação:
#SE: sqrt(MSE*((1/n)+ ((Xbarra)^2)/Sxx)
SE.beta0 <- sqrt(var_hat$MSE*((1/n)+ ((Xbarra)^2)/Sxx))
SE.beta0

# 4####################################################  INTERVALO DE CONFIANÇA PARA BETA0:
#IC.beta0 superior: beta0_hat + t(n-2; alfa/2) * SE
#IC.beta0 inferior: beta0_hat - t(n-2; alfa/2) * SE
gl <- n-2
alfa <- 0.05
tcritico <- qt(1-alfa/2, df = gl)

IC.beta0.superior <- coeficientes$beta0_hat + tcritico * SE.beta0
IC.beta0.inferior <- coeficientes$beta0_hat - tcritico * SE.beta0

# 5### Calcular o Standard Error (SE) para beta1 em cada simulação:
#SE: sqrt(MSE*((1/n)+ ((Xbarra)^2)/Sxx)
SE.beta1 <- sqrt((var_hat$MSE)*(1/Sxx))
SE.beta1

# 6#################################################### INTERVALO DE CONFIANÇA PARA BETA1:
IC.beta1.superior <- coeficientes$beta1_hat + tcritico * SE.beta1
IC.beta1.inferior <- coeficientes$beta1_hat - tcritico * SE.beta1

resultado.final <- data.frame(
  beta0_hat = coeficientes$beta0_hat,
  beta1_hat = coeficientes$beta1_hat,
  SSE = var_hat$SSE,
  MSE = var_hat$MSE,
  IC.beta0.upper = IC.beta0.superior,
  IC.beta0.lower = IC.beta0.inferior,
  IC.beta1.upper = IC.beta1.superior,
  IC.beta1.lower = IC.beta1.inferior
)

resultado.final


#################################################### TESTES DE HIPÓTESES PARA OS BETAS
###########################Teste para beta0
#Manualmente:
#H0: beta0 = 5
#H1: beta0 != 5
#alfa: 5%
#estatistica do teste = t = (beta0_hat - beta0)/sqrt(MSE*((1/n)+ ((Xbarra)^2)/Sxx) ~ t(gl=n-2)
#p-valor
tcalc.beta0 <- (resultado.final$beta0_hat - 5)/SE.beta0
tcritico
p.valor.beta0 <- 2*pt(-abs(tcalc.beta0), lower.tail = TRUE, df = gl) #probabilidade acumulada na cauda inferior
p.valor.beta0
rejeitaH0.beta0 <- ifelse(p.valor.beta0 < 0.05, "Rejeita H0", "Não rejeita H0")
###########################Teste para beta1
#H0: beta1 = 2
#H1: beta1 != 2
#alfa: 5%
tcalc.beta1 <- (resultado.final$beta1_hat - 2)/SE.beta1
tcritico
p.valor.beta1 <- 2*pt(-abs(tcalc.beta1), lower.tail = TRUE, df = gl) #probabilidade acumulada na cauda inferior
p.valor.beta1
rejeitaH0.beta1 <- ifelse(p.valor.beta1 < 0.05, "Rejeita H0", "Não rejeita H0")
#H0: beta1 = 1.8
#H1: beta1 != 1.8
#alfa: 5%
tcalc.beta1.2 <- (resultado.final$beta1_hat - 1.8)/SE.beta1
tcritico
p.valor.beta1.2 <- 2*pt(-abs(tcalc.beta1.2), lower.tail = TRUE, df = gl) #probabilidade acumulada na cauda inferior
p.valor.beta1.2
rejeitaH0.beta1.2 <- ifelse(p.valor.beta1.2 < 0.05, "Rejeita H0", "Não rejeita H0")

resultado.final <- data.frame(
  beta0_hat = coeficientes$beta0_hat,
  beta1_hat = coeficientes$beta1_hat,
  SSE = var_hat$SSE,
  MSE = var_hat$MSE,
  IC.beta0.upper = IC.beta0.superior,
  IC.beta0.lower = IC.beta0.inferior,
  IC.beta1.upper = IC.beta1.superior,
  IC.beta1.lower = IC.beta1.inferior,
  pvalor.beta0.H0.5 = p.valor.beta0,
  rejeitaH0.beta0 = rejeitaH0.beta0,
  pvalor.beta1.H0.2 = p.valor.beta1,
  rejeitaH0.beta1 = rejeitaH0.beta1,
  pvalor.beta1.H0.1.8 = p.valor.beta1.2,
  rejeitaH0.beta1.2 = rejeitaH0.beta1.2
)

resultado.final
table(resultado.final$rejeitaH0.beta0)
table(resultado.final$rejeitaH0.beta1)
table(resultado.final$rejeitaH0.beta1.2)


#################################################### Gráficos
### Distribuição do beta0_hat:
#Estatísticas
media.beta0_hat <- mean(resultado.final$beta0_hat)
var.beta0_hat <- var(resultado.final$beta0_hat)
#Grafico
library(ggplot2)
distribuicao.beta0 <- ggplot(resultado.final, aes(x = beta0_hat)) + 
  geom_histogram(aes(y = ..density..), 
                 bins = 30, 
                 fill = "gray", 
                 color = "white", 
                 alpha = 0.8) +
  geom_density(color = "#1f4e79", size = 1) +
  geom_vline(aes(xintercept = mean(beta0_hat, na.rm = TRUE)), 
             linetype = "dashed", 
             color = "red", 
             linewidth = 1) +
  labs(title = "Distribuição das Estimativas de β0",x = "Estimativas de β0",y = "Densidade") +
  scale_y_continuous(expand = c(0, 0)) +
  annotate("text", x = media.beta0_hat, y = 0.02, label = paste0("Média = ", round(media.beta0_hat, 3)), color = "red", hjust = 0) +
  annotate("text", x = media.beta0_hat, y = 0.05, label = paste0("Variância = ", round(var.beta0_hat, 3)), color = "red", hjust = 0) +
  theme(axis.line = element_line(linewidth=0.5, colour="black"), panel.background = element_rect(fill=NA))+
  theme(plot.title = element_text(face = "bold", hjust = 0.5), panel.grid.major = element_line(color = "gray90"))

distribuicao.beta0

#ggsave("fig.png", plot = last_plot(), device = "png", path = NULL,
#scale = 2, width = 5, height = 3.6, units = "in",
#dpi = 600, limitsize = TRUE)

### Distribuição do beta1_hat: 
#Estatísticas
media.beta1_hat <- mean(resultado.final$beta1_hat)
var.beta1_hat <- var(resultado.final$beta1_hat)

library(ggplot2)
distribuicao.beta1 <- ggplot(resultado.final, aes(x = beta1_hat)) + 
  geom_histogram(aes(y = ..density..), 
                 bins = 30, 
                 fill = "gray", 
                 color = "white", 
                 alpha = 0.8) +
  geom_density(color = "#1f4e79", size = 1) +
  geom_vline(aes(xintercept = mean(beta1_hat, na.rm = TRUE)), 
             linetype = "dashed", 
             color = "red", 
             linewidth = 1) +
  labs(title = "Distribuição das Estimativas de β1",x = "Estimativas de β1",y = "Densidade") +
  scale_y_continuous(expand = c(0, 0)) +
  annotate("text", x = media.beta1_hat, y = 0.07, label = paste0("Média = ", round(media.beta1_hat, 4)), color = "red", hjust = 0) +
  annotate("text", x = media.beta1_hat, y = 0.15, label = paste0("Variância = ", round(var.beta1_hat, 3)), color = "red", hjust = 0) +
  theme(axis.line = element_line(linewidth=0.5, colour="black"), panel.background = element_rect(fill=NA))+
  theme(plot.title = element_text(face = "bold", hjust = 0.5),panel.grid.major = element_line(color = "gray90"))

distribuicao.beta1



#################################################### Contagem dos intervalos de confiança
# Quantos intervalos para beta0 contém o valor 5?
resultado.final %>% filter(IC.beta0.upper >= 5 & IC.beta0.lower <=5 ) %>% nrow()
# Quantos intervalos para beta1 contém o valor 2?
resultado.final %>% filter(IC.beta1.upper >= 2 & IC.beta1.lower <=2 ) %>% nrow()
# Quantos intervalos para beta0 e beta1 juntos?
resultado.final %>% filter(IC.beta0.upper >= 5 & IC.beta0.lower <=5 & IC.beta1.upper >= 2 & IC.beta1.lower <=2) %>% nrow()



#################################################### Contagem de rejeições
library(dplyr)
# 4. 
table(resultado.final$rejeitaH0.beta0)
#5.
table(resultado.final$rejeitaH0.beta1)
#6.
table(resultado.final$rejeitaH0.beta1.2)
#7
resultado.final %>% filter(rejeitaH0.beta0 == "Rejeita HO"| rejeitaH0.beta1 == "Rejeita H0") %>% nrow()


