library(TCGAbiolinks)
# Aligned against Hg19
query.exp.hg19 <- GDCquery(project = "TCGA-GBM",
                           data.category = "Gene expression",
                           data.type = "Gene expression quantification",
                           platform = "Illumina HiSeq", 
                           file.type  = "normalized_results",
                           experimental.strategy = "RNA-Seq",
                           legacy = TRUE)
GDCdownload(query.exp.hg19)
data <- GDCprepare(query.exp.hg19)

query.exp.hg38 <- GDCquery(project = "TCGA-GBM", 
                           data.category = "Transcriptome Profiling", 
                           data.type = "Gene Expression Quantification", 
                           workflow.type = "HTSeq - FPKM",
GDCdownload(query.exp.hg38)

query.exp.hg38_1 <- GDCquery(project = "TCGA-BLCA", 
                           data.category = "Transcriptome Profiling", 
                           data.type = "Gene Expression Quantification", 
                           workflow.type = "HTSeq - Counts",
                           sample.type = "Solid Tissue Normal")

query.exp.hg38_1 <- GDCquery(project = "TCGA-LUAD", 
                             data.category = "Transcriptome Profiling", 
                             data.type = "Gene Expression Quantification", 
                             workflow.type = "HTSeq - Counts")

GDCdownload(query.exp.hg38_1)

