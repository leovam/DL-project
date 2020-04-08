# all available tumor types in TCGA data base
tumor_type_1 <- c('LAML', 'ACC', 'BLCA', 'LGG', 'BRCA', 'CESC', 'CHOL', 'LCML', 
              'COAD', 'CNTL', 'ESCA','FPPP','GBM', 'HNSC', 'KICH', 'KIRC', 
              'KIRP', 'LIHC', 'LUAD', 'LUSC', 'DLBC', 'MESO', 'MISC', 'OV',
              'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT',
              'THYM', 'THCA', 'UCS', 'UCEC', 'UVM')

# tumor types in the paper
tumor_type_2 <- c('ACC', 'BLCA', 'LGG', 'BRCA', 'CESC', 'CHOL', 
                'COAD', 'ESCA','GBM', 'HNSC', 'KICH', 'KIRC', 
                'KIRP', 'LAML', 'LUAD', 'LIHC', 'LUSC', 'DLBC', 'MESO', 
                'OV',   'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 
                'STAD', 'TGCT', 'THYM', 'THCA', 'UCS', 'UCEC', 'UVM')


for (tumor in tumor_type_2){
  query.exp.hg38_1 <- GDCquery(project = paste('TCGA', tumor, sep = '-'), 
                               data.category = "Transcriptome Profiling", 
                               data.type = "Gene Expression Quantification", 
                               workflow.type = "HTSeq - Counts")
  
  GDCdownload(query.exp.hg38_1)
}
