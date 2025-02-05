library("SEraster")
library(SpatialExperiment)
load("/Users/zhezhenwang/Documents/patro/data/merfish_mousePOA.rda")
#dim: 155 6509 

rastGexp <- SEraster::rasterizeGeneExpression(merfish_mousePOA, 
                                              assay_name="volnorm", 
                                              resolution = 50)

tmp = as.data.frame(spatialCoords(merfish_mousePOA))
dim(tmp) #[1] 6509    2
img = as.matrix(assay(merfish_mousePOA))
identical(row.names(tmp),colnames(img)) #[1] TRUE
alldf = cbind(tmp,t(img))
dim(alldf) #[1] 6509  157
#write.csv(tmp,file = "merfish6k.csv")

seraster = spatialCoords(rastGexp)
dim(seraster) #[1] 1301    2

sorted_df <- alldf[order(alldf$y, alldf$x), ]
#ace2 = sorted_df[,1:3]
#block_mean = c(5122, 5119, 5116, 3562, 3451, 2650, 1903, 766)
#error_mean = c(0.0, 0.0, 0.0, 0.008, 0.014, 0.023, 0.032, 0.041)
## construct tree w/ all genes
block_mean = c(12424, 6268, 6268, 4285, 3040, 1117)
error_mean = c(0.08244909607714947, 0.07420418646943452, 0.06745835133584957, 
               0.061836822057862104, 0.05708014343802656, 0.06033075389023121, 
               0.06334723552378331, 0.06844562482317873, 0.07413721146691653, 
               0.07845744323997668)

#block_med = c(5122, 5119, 5116, 3562, 3514, 3460, 3433, 3382)
#seblock = c(18.95,18.94,18.93,27.54,28.11,33.5,40.74,65)
#seblock = c(18.95,18.94,18.93,27.54,27.76,28.1,28.13,28.54)
seblock = c(18.95,10.2,10.2,23.47,30.64,54)
for (i in seblock){
  rastGexp <- SEraster::rasterizeGeneExpression(merfish_mousePOA, 
                                                assay_name="volnorm", 
                                                resolution = i)
  seraster = spatialCoords(rastGexp)
  tmp = split(as.data.frame(seraster),seraster[,2])
  errors_all = c()
  errors = c()
  for(j in 1:nrow(ace2)){
    for(y in names(tmp)){
      inity = tmp[[y]][,2][1]
      if(ace2[j,2]>inity & ace2[j,2]<inity+i){
        initx = tmp[[y]][,1][1]
        for(x in y:(nrow(tmp[[y]])-1)){
          if(ace2[j,1]>initx & ace2[j,1]<initx + i){
            #print(ace2[j,3])
            errors = c(errors,ace2[j,3])
            #print(errors)
          }else{
            if(length(errors>0)){
              score = (max(abs(errors - median(errors))))/(max(errors) - min(errors)+0.01)
              errors_all = c(errors_all,score)
              errors = c()
              }
            }
          }
      }else{
        if(length(errors>0)){
          # if(max(errors) == min(errors)){
          #   divider = max(errors) - min(errors)+0.0001
          # }else{
          #   divider = max(errors) - min(errors)
          # }
          score = (max(abs(errors - median(errors))))/(max(errors) - min(errors)+0.01)
          errors_all = c(errors_all,score)
          errors_all = c(errors_all,errors)
          errors = c()
        }
      }
    }  
  }
  print(mean(errors_all))
  #print(max(errors_all))
}

findblocks = function(){
  for (i in seblock){
    rastGexp <- SEraster::rasterizeGeneExpression(merfish_mousePOA,
                                                  assay_name="volnorm",
                                                  resolution = i)
    seraster = spatialCoords(rastGexp)
    tmp = split(as.data.frame(seraster),seraster[,2])
    errors_all = c()
    errors = c()
    for(j in 1:nrow(ace2)){
      for(y in names(tmp)){
        inity = tmp[[y]][,2][1]
        if(ace2[j,2]>inity & ace2[j,2]<inity+i){
          initx = tmp[[y]][,1][1]
          for(x in y:(nrow(tmp[[y]])-1)){
            if(ace2[j,1]>initx & ace2[j,1]<initx + i){
              #print(ace2[j,3])
              errors = c(errors,ace2[j,3])
              #print(errors)
            }else{
              if(length(errors>0)){
                score = (max(abs(errors - median(errors))))/(max(errors) - min(errors)+0.01)
                errors_all = c(errors_all,score)
                errors = c()
              }
            }
          }
        }else{
          if(length(errors>0)){
            # if(max(errors) == min(errors)){
            #   divider = max(errors) - min(errors)+0.0001
            # }else{
            #   divider = max(errors) - min(errors)
            # }
            score = (max(errors) - median(errors))/(max(errors) - min(errors)+0.01)
            errors_all = c(errors_all,score)
            errors_all = c(errors_all,errors)
            errors = c()
          }
        }
      }
    }
    if(mean(errors_all)<given){
      
    }
  }
}

## nnSVG
library("nnSVG")
ori = nnSVG(merfish_mousePOA,assay_name="volnorm")
poa_se50 = SEraster::rasterizeGeneExpression(merfish_mousePOA, 
                                  assay_name="volnorm", 
                                  resolution = 50)
ser50 <- nnSVG(poa_se50,assay_name="pixelval",order = "Sum_coords")
poa_se100 = SEraster::rasterizeGeneExpression(merfish_mousePOA, 
                                             assay_name="volnorm", 
                                             resolution = 100)
ser100 <- nnSVG(poa_se100,assay_name="pixelval",order = "Sum_coords")
poa_se200 = SEraster::rasterizeGeneExpression(merfish_mousePOA, 
                                              assay_name="volnorm", 
                                              resolution = 200)
ser200 <- nnSVG(poa_se200,assay_name="pixelval",order = "Sum_coords")
poa_se400 = SEraster::rasterizeGeneExpression(merfish_mousePOA, 
                                              assay_name="volnorm", 
                                              resolution = 400)
ser400 <- nnSVG(poa_se400,assay_name="pixelval",order = "Sum_coords")




library("scran")
# counts = read.csv("/Users/zhezhenwang/Documents/patro/data/datasets-mouse_brain_map-BrainReceptorShowcase-Slice2-Replicate1-cell_by_gene_S2R1.csv",row.names = 1)
# metadata = read.csv("/Users/zhezhenwang/Documents/patro/data/datasets-mouse_brain_map-BrainReceptorShowcase-Slice2-Replicate1-cell_metadata_S2R1.csv",row.names = 1)
# subcounts = counts[,!grepl("Blank",colnames(counts))]
# spe <- SpatialExperiment::SpatialExperiment(
#   assays = list(counts = t(subcounts)),
#   spatialCoords = as.matrix(metadata[,c("center_x","center_y")]),
# )
# #spe <- filter_genes(spe)
# spe <- computeLibraryFactors(spe)
# spe <- logNormCounts(spe)
# ori <- nnSVG(spe)
# tmp = SEraster::rasterizeGeneExpression(spe, 
#                                   assay_name="logcounts", 
#                                   resolution = 50)
# ser50 <- nnSVG(tmp,assay_name="pixelval")
# tmp = SEraster::rasterizeGeneExpression(spe, 
#                                         assay_name="logcounts", 
#                                         resolution = 100)
# brainser100 <- nnSVG(tmp,assay_name="pixelval")

# set seed for reproducibility
set.seed(123)
# using a single thread in this example
#spe <- spe[, colData(merfish_mousePOA)$in_tissue == 1]
# spe <- filter_genes(spe)
# spe <- computeLibraryFactors(spe)
# spe <- logNormCounts(spe)


# file_path = "/Users/zhezhenwang/Documents/patro/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"
# tmp = read.csv(file_path)
# dim(tmp)
# [1] 1,027,848     170

# library(nnSVG)

file_path = "/Users/zhezhenwang/Documents/patro/merfish6k.csv"
poa = read.csv(file_path,row.names = 1)
poa <- SpatialExperiment::SpatialExperiment(
  assays = list(counts = t(poa[,-c(1,2)])),
  spatialCoords = as.matrix(poa[,c("x","y")]),
)
poa <- computeLibraryFactors(poa)
poa <- logNormCounts(poa)
ori <- nnSVG(poa)

# qdtree = read.csv("~/Documents/patro/quadtreedf.csv")
# qdtree = read.csv("~/Documents/patro/quadtreedf_mean0.5.csv")
qdtree = read.csv("~/Documents/patro/quadtreedf_meanmean0.7.csv")
qdtree <- SpatialExperiment::SpatialExperiment(
  assays = list(counts = t(qdtree[,-c(1,2)])),
  spatialCoords = as.matrix(qdtree[,c("x","y")]),
)
qdtree <- computeLibraryFactors(qdtree)
qdtree <- logNormCounts(qdtree)
qdtree <- nnSVG(qdtree)

genes = intersect(row.names(rowData(qdtree)),row.names(rowData(ori)))
plot(rowData(qdtree)[genes,"rank"],rowData(ori)[genes,"rank"],
  col = "blue", pch = 16,  # Solid blue dots
  main = "thereshold 0.7",
  xlab = "qdtree rank", ylab = "single cell rank"
)

# Add correlation legend
legend(
  "topleft", 
  legend = paste("Correlation:", round(cor(rowData(qdtree)[genes,"rank"],
                                           rowData(ori)[genes,"rank"],
                                           method = "spearman"), 2)),
  bty = "n", cex = 0.9, text.col = "blue"
)
dev.copy2pdf(file = "cor_quadtree_meanmean0.7.pdf")

seraster = SEraster::rasterizeGeneExpression(poa, 
                                        assay_name="logcounts", 
                                        resolution = 23.47)
seraster <- nnSVG(seraster,assay_name="pixelval")
#genes = intersect(row.names(rowData(seraster)),row.names(rowData(ori)))
plot(rowData(seraster)[,"rank"],rowData(ori)[,"rank"],
     col = "darkred", pch = 16,  # Solid blue dots
     main = "same # of blocks as quadtree threshold 0.7",
     xlab = "SEraster rank", ylab = "single cell rank"
)

# Add correlation legend
legend(
  "topleft", 
  legend = paste("Correlation:", round(cor(rowData(seraster)[genes,"rank"],
                                           rowData(ori)[genes,"rank"],
                                           method = "spearman"), 2)),
  bty = "n", cex = 0.9, text.col = "darkred"
)

dev.copy2pdf(file = "cor_seraster_meanmean0.7.pdf")



