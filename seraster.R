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

findblocks2 = function(target, start = 0, end = 6509){
  while(start < end){
    mid = (start+ end) / 2
    rastGexp <- SEraster::rasterizeGeneExpression(merfish_mousePOA,
                                                  assay_name="volnorm",
                                                  resolution = mid)
    # print(mid)
    # print(tmp)
    # print(paste0("target",target))
    seraster = lapply(rastGexp, function(x) colData(x)$cellID_list)
    errors <- lapply(seraster, function(x) {
      sapply(x, function(pixel) {
        if (length(pixel) > 1) {  
          max(sapply(pixel, function(cell) quantile(abs(countsdf[,cell] - apply(countsdf[,pixel],1,mean)) / 
                                                      (maxvec - minvec + 0.01),prob= 1)))
        } else {
          0
        }
      })
    })
    if(round(end,2) == round(mid,2) | round(start,2) == round(mid,2)) break
    if(errors<target){
      end = mid
    }else if(errors == target){
      break
    }else{
      start = mid 
    }
  }
  seraster = spatialCoords(rastGexp)
  tmp = nrow(seraster)
  return(tmp)
}

seblock = c(18.95,10.2,10.2,23.47,30.64,54)

findblocks = function(target, start = 0, end = 6509){
  while(start < end){
    mid = (start+ end) / 2
    rastGexp <- SEraster::rasterizeGeneExpression(merfish_mousePOA,
                                                  assay_name="volnorm",
                                                  resolution = mid)
    seraster = spatialCoords(rastGexp)
    tmp = nrow(seraster)
    print(mid)
    print(tmp)
    print(paste0("target",target))
    if(round(end,2) == round(mid,2) | round(start,2) == round(mid,2)) break
    if(tmp<target){
        end = mid
    }else if(tmp == target){
      break
    }else{
      start = mid 
    }
  }
}

seblock = c(1,6.8, 14.11034,21.03906,26.3125,32.43506,39.18213,48.10059,58.66211,82.49512)
countsdf = assay(merfish_mousePOA)
#for (i in seblock){
  rastGexp <- lapply(seblock, function(i) SEraster::rasterizeGeneExpression(merfish_mousePOA, 
                                                assay_name="volnorm", 
                                                resolution = i))
  seraster = lapply(rastGexp, function(x) colData(x)$cellID_list)
#spatialCoords(rastGexp)
  #tmp = split(as.data.frame(seraster),seraster[,2])
  # errors = lapply(seraster, function(pixel) sapply(pixel, function(cell)
  #   max(abs(countsdf[,cell] - mean(countsdf[,cell])))/ (max(countsdf[,cell]) - min(countsdf[,cell]) + 0.01)))
  maxvec = apply(countsdf,1,max)
  minvec = apply(countsdf,1,min)
  errors <- lapply(seraster, function(x) {
    sapply(x, function(pixel) {
      if (length(pixel) > 1) {  
        max(sapply(pixel, function(cell) quantile(abs(countsdf[,cell] - apply(countsdf[,pixel],1,mean)) / 
                 (maxvec - minvec + 0.01),prob= 1)))
      } else {
        0
      }
    })
  })
    #hist(errors,100)
#}
boxplot(errors)
names(errors) = c(6509, 6477, 5841, 4743, 3764, 2798, 2022, 1385, 933, 510)
df <- bind_rows(lapply(errors, function(x) data.frame(errors = x)), .id = "blocks")
df$methods = 'SEraster'

library(jsonlite)
r_list <- fromJSON("/Users/zhezhenwang/Documents/patro/errorlist.json")
#print(r_list)
names(r_list) = c(6509, 6477, 5841, 4743, 3764, 2798, 2022, 1385, 933, 510)
pydf <- bind_rows(lapply(r_list, function(x) data.frame(errors = x)), .id = "blocks")
pydf$methods = 'quadtree'

toplot = rbind(df,pydf)
toplot$blocks <- factor(toplot$blocks, levels = c(6509, 6477, 5841, 4743, 3764, 2798, 2022, 1385, 933, 510))
library(ggrain)

pdf(file = "rain_cloud_cutoffs.pdf")
for(i in levels(toplot$blocks)){
  print(i)
  subtoplot = subset(toplot,blocks == i)
  p = ggplot(subtoplot, aes(blocks, errors, fill = methods, color = methods)) +
       geom_rain(alpha = .5,
                 boxplot.args.pos = list(
                   position = ggpp::position_dodgenudge(x = .1, width = 0.1), width = 0.1
                 )) +
       #geom_boxplot(width = 0.15, alpha = 0.7) +
       theme_classic() +
       scale_fill_manual(values=c("dodgerblue", "darkorange"))+
       scale_color_manual(values=c("dodgerblue", "darkorange"))
  print(p)
}
dev.off()  

#dev.copy2pdf(file = 'raincloud.pdf',height = 5, width = 20)

# colnames(df) <- c("errors", "Group")
# for (i in seblock){
#   rastGexp <- SEraster::rasterizeGeneExpression(merfish_mousePOA, 
#                                                 assay_name="volnorm", 
#                                                 resolution = i)
#   seraster = spatialCoords(rastGexp)
#   tmp = split(as.data.frame(seraster),seraster[,2])
#   errors_all = c()
#   errors = c()
#   for( x in 1: ncol(sorted_df)){
#   for(j in 1:nrow(ace2)){
#     for(y in names(tmp)){
#       inity = tmp[[y]][,2][1]
#       if(ace2[j,2]>inity & ace2[j,2]<inity+i){
#         initx = tmp[[y]][,1][1]
#         for(x in y:(nrow(tmp[[y]])-1)){
#           if(ace2[j,1]>initx & ace2[j,1]<initx + i){
#             errors = c(errors,sorted_df[j,x])
#             #print(sorted_df[j,x])
#           }else{
#             if(length(errors>0)){
#               score = (max(abs(errors - mean(errors))))/(max(errors) - min(errors)+0.01)
#               if(score>1){
#                 print(max(abs(errors - mean(errors))))
#                 print(max(errors) - min(errors)+0.01)
#               }
#               errors_all = c(errors_all,score)
#               errors = c()
#               }
#             }
#         }
#       }else{
#         if(length(errors>0)){
#           # if(max(errors) == min(errors)){
#           #   divider = max(errors) - min(errors)+0.0001
#           # }else{
#           #   divider = max(errors) - min(errors)
#           # }
#           score = max(abs(errors - mean(errors)))/(max(errors) - min(errors)+0.01)
#           if(score>1){
#             print(max(abs(errors - mean(errors))))
#             print(max(errors) - min(errors)+0.01)
#           }
#           errors_all = c(errors_all,score)
#           errors_all = c(errors_all,errors)
#           errors = c()
#         }
#       }
#     }  
#   }
 # print(max(errors_all))
  #print(max(errors_all))
#}
#}

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
            #errors_all = c(errors_all,errors)
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
qdtree = read.csv("~/Documents/patro/quadtreedf_max0.5.csv")
colnames(qdtree) = str_split_fixed(colnames(qdtree),"[.]",3)[,2]
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
  main = "thereshold 0.5",
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
dev.copy2pdf(file = "cor_quadtree_max0.5.pdf")

seraster = SEraster::rasterizeGeneExpression(poa, 
                                        assay_name="logcounts", 
                                        resolution = 32.43506)
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

dev.copy2pdf(file = "cor_seraster_max0.5.pdf")


# remotes::install_github('jorvlan/raincloudplots')
# library(ggplot2)
# library(raincloudplots)
# install.packages('ggrain')






