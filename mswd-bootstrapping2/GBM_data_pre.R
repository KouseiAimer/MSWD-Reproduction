
library(cgdsr)
library(rjson)
############### download data ##########################
# prognostic genes
json.data = fromJSON(file="GBM/prognostic_glioma.json")
genes = sapply(json.data, function(x)x$Gene)

# Create CGDS object
mycgds = CGDS("http://www.cbioportal.org/")
# Get list of cancer studies at server
studies = getCancerStudies(mycgds)
str(studies)
idx = which(grepl("Brain Lower Grade Glioma", studies$name, fixed = TRUE) )	#Glioma
studies$name[idx]
studies$cancer_study_id[idx]
#"Brain Lower Grade Glioma (TCGA, Firehose Legacy)" 		"lgg_tcga"
id = "lgg_tcga"
studies$description[idx[which(studies$cancer_study_id[idx]==id)]]

# case list
mycaselist = getCaseLists(mycgds, id)
list = mycaselist$case_list_id[1] # all samples

# profile for genes
mygeneticprofile = getGeneticProfiles(mycgds, id)
profile = mygeneticprofile[6,1]
print(mygeneticprofile[6,2]) # Methylation (HM450)

# download data
data.tmp = getProfileData(mycgds, genes, profile, list)
# colnames(data.tmp)
# remove the subjects and genes with NA
is_all_na_in_columns <- apply(data.tmp, 2, function(x){all(is.na(x))})
sum(is_all_na_in_columns)
data <- data.tmp[,-which(is_all_na_in_columns)]
is_any_na_in_rows <- apply(data, 1, function(x){any(is.na(x))})
sum(is_any_na_in_rows)
if(sum(is_any_na_in_rows)==0){
  data_low <- data
}else{
  data_low <- data[-which(is_any_na_in_rows),]
}

idx = which(grepl("Glioblastoma Multiforme", studies$name, fixed = TRUE) )	#Glioblastoma
studies$name[idx]
studies$cancer_study_id[idx]
# "Glioblastoma Multiforme (TCGA, Firehose Legacy)" 		"gbm_tcga"
id = "gbm_tcga"
studies$description[idx[which(studies$cancer_study_id[idx]==id)]]

# case list
mycaselist = getCaseLists(mycgds, id)
list = mycaselist$case_list_id[1] # all samples

# profile for genes
mygeneticprofile = getGeneticProfiles(mycgds, id)
profile = mygeneticprofile[7,1]
print(mygeneticprofile[7,2]) # Methylation (HM450)

# download data
data.tmp = getProfileData(mycgds, genes, profile, list)
# colnames(data.tmp)
# remove the subjects and genes with NA
is_all_na_in_columns <- apply(data.tmp, 2, function(x){all(is.na(x))})
sum(is_all_na_in_columns)
data <- data.tmp[,-which(is_all_na_in_columns)]
is_any_na_in_rows <- apply(data, 1, function(x){any(is.na(x))})
sum(is_any_na_in_rows)
if(sum(is_any_na_in_rows)==0){
  data_high <- data
}else{
  data_high <- data[-which(is_any_na_in_rows),]
}

genes_both = intersect(colnames(data_low), colnames(data_high))
data_low = data_low[,which(colnames(data_low)%in%genes_both)]
data_high = data_high[,which(colnames(data_high)%in%genes_both)]
all(colnames(data_low)==colnames(data_high))

# write.table(colnames(data_low), file='GBM/GBM_used_prognostic_genes_names.txt', sep=' ', row.names = F, col.names = F)
# write.table(data_low, file='GBM/GBM_prognostic_low.txt', sep=' ', row.names = F, col.names = F)
# write.table(data_high, file='GBM/GBM_prognostic_high.txt', sep=' ', row.names = F, col.names = F)

############################ END ###########################################

################## investigate some genes #########################
library(rjson)
used_genes = colnames(data_low)

# GO terms gene sets
json.go.data = fromJSON(file='GBM/c5.go.bp.v2023.1.Hs.json')

############### select some specific GO terms ################

genes_tmp <- intersect(used_genes, json.go.data[[2631]]$geneSymbols) # GO:0000278 MITOTIC_CELL_CYCLE
idx = which(used_genes%in%genes_tmp)
write.table(idx, file='GBM/idx_genes_mitotic_cell_cycle.txt', sep=' ', row.names = F, col.names = F)

genes_tmp <- intersect(used_genes, json.go.data[[2634]]$geneSymbols) # GO:1903047 MITOTIC_CELL_CYCLE_PROCESS
idx = which(used_genes%in%genes_tmp)
write.table(idx, file='GBM/idx_genes_mitotic_cell_cycle_process.txt', sep=' ', row.names = F, col.names = F)

genes_tmp <- intersect(used_genes, json.go.data[[595]]$geneSymbols) # GO:0022402 CELL_CYCLE_PROCESS 
idx = which(used_genes%in%genes_tmp)
write.table(idx, file='GBM/idx_genes_cell_cycle_process.txt', sep=' ', row.names = F, col.names = F)

genes_tmp <- intersect(used_genes, json.go.data[[588]]$geneSymbols) # GO:0007049 CELL_CYCLE 
idx = which(used_genes%in%genes_tmp)
write.table(idx, file='GBM/idx_genes_cell_cycle.txt', sep=' ', row.names = F, col.names = F)

genes_tmp <- intersect(used_genes, json.go.data[[1202]]$geneSymbols) # GO:0006259 DNA_METABOLIC_PROCESS
idx = which(used_genes%in%genes_tmp)
write.table(idx, file='GBM/idx_genes_DNA_metabolic_process.txt', sep=' ', row.names = F, col.names = F)

genes_tmp <- intersect(used_genes, json.go.data[[604]]$geneSymbols) # GO:0051301 CELL_DIVISION 
idx = which(used_genes%in%genes_tmp)
write.table(idx, file='GBM/idx_genes_cell_division.txt', sep=' ', row.names = F, col.names = F)

genes_tmp <- intersect(used_genes, json.go.data[[2646]]$geneSymbols) # GO:0140014 MITOTIC_NUCLEAR_DIVISION
idx = which(used_genes%in%genes_tmp)
write.table(idx, file='GBM/idx_genes_mitotic_nuclear_division.txt', sep=' ', row.names = F, col.names = F)

