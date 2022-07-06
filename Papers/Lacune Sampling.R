library(AnalyzeFMRI)
library(stringr)
library(crayon)

set.seed(1)

# Training Valid Test Split -----------------------------------------------

list.t1 <- list.files("MAS_W2/T1softTiss/")
list.flair <- list.files("MAS_W2/FLAIRinT1space/")
list.lacune <- list.files("MAS_W2/lacune_T1space/")

list.id <- str_extract(list.t1, "^[0-9]{4}")

# id of lacune samples
list.id.lacune <- str_extract(list.lacune, "^[0-9]{4}")
# id of nonlacune samples
list.id.nonlacune <- setdiff(list.id, list.id.lacune)

# Split id of lacune and nonlacune samples into three sets
list.id.lacune.sets <- list(list.id.lacune[1:18], list.id.lacune[19:27], list.id.lacune[28:35])
list.id.nonlacune.sets <- list(list.id.nonlacune[1:188], list.id.nonlacune[189:282],
                               list.id.nonlacune[283:376])


# Train - Positive --------------------------------------------------------
# Train/valid/test index
tvt <- 1
# For each set, generate positive samples + flipped samples
max.rows <- 6000
data.train.lacunes <- array(NA, dim = c(max.rows, 5208))
i <- 1
for (id in list.id.lacune.sets[[tvt]]) {
   cat(white$bgBlack(paste("Processing",id,"\n")))
   
   file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
   soft <- f.read.nifti.volume(file.soft)
   
   file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
   t1 <- f.read.nifti.volume(file.t1)
   
   file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
   flair <- f.read.nifti.volume(file.flair)
   
   file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
   lacune <- f.read.nifti.volume(file.lacune)
   
   for (x in 1:dim(soft)[1]) {
      for (y in 1:dim(soft)[2]) {
         for (z in 1:dim(soft)[3]) {
            if (lacune[x,y,z,1] == 0) next
            print(paste("Lacune at [", x, y, z, "]"))
            data.train.lacunes[i, 1] <- as.numeric(id)
            data.train.lacunes[i, 2] <- x
            data.train.lacunes[i, 3] <- y
            data.train.lacunes[i, 4] <- z
            
            patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
            patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
            
            data.train.lacunes[i, 5:2605] <- patch.t1
            data.train.lacunes[i, 2606:5206] <- patch.flair
            
            data.train.lacunes[i, 5207] <- 1
            data.train.lacunes[i, 5208] <- 0
            
            i <- i + 1
            if (i > max.rows) break
         }
      }
   }
   
}
# Remove unfilled rows in array
numrows <- max(which(!is.na(data.train.lacunes[,1])))
data.train.lacunes <- data.train.lacunes[1:numrows,]

# Train - Negative ----------------------------------------------------------

max.rows2 <- 50000
data.train.nonlacunes <- array(NA, dim = c(max.rows2, 5208))
i <- 1
for (id in list.id.lacune.sets[[tvt]]) {
   cat(white$bgBlack(paste("Processing",id,"\n")))
   
   file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
   soft <- f.read.nifti.volume(file.soft)
   
   file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
   t1 <- f.read.nifti.volume(file.t1)
   
   file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
   flair <- f.read.nifti.volume(file.flair)
   
   file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
   if (file_test("-f", file.lacune)) {
      lacune <- f.read.nifti.volume(file.lacune)
   } else {
      lacune <- array(data = 0, dim = dim(soft))
   }
   
   # Take every kth pixel of matter. Skip if not matter or if a lacune. Sequence starts randomly between 26 and 76
   
   for (x in seq(round(runif(1, 26, 56)), dim(soft)[1] - 26, by = 15)) {
      for (y in seq(round(runif(1, 26, 56)), dim(soft)[2] - 26, by = 15)) {
         for (z in seq(round(runif(1, 26, 56)), dim(soft)[3] - 26, by = 15)) {
            
            # Isolate a 5x5 square in the middle of the sample. If the whole square is 0, skip
            midregion <- sum(soft[(x-4):(x+4), (y-4):(y+4), (z-4):(z+4),1])
            # Skip if pixel is not in brain matter, or is a lacune
            if (lacune[x,y,z,1] == 1 | midregion == 0) next
            
            print(paste("Non-lacune at [", x, y, z, "]"))
            data.train.nonlacunes[i, 1] <- as.numeric(id)
            data.train.nonlacunes[i, 2] <- x
            data.train.nonlacunes[i, 3] <- y
            data.train.nonlacunes[i, 4] <- z
            
            patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
            patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
            
            data.train.nonlacunes[i, 5:2605] <- patch.t1
            data.train.nonlacunes[i, 2606:5206] <- patch.flair
            
            data.train.nonlacunes[i, 5207] <- 0
            data.train.nonlacunes[i, 5208] <- 1
            
            i <- i + 1
            if (i > max.rows2) {
               stop(paste("Reached max number of rows", max.rows2))
            }
         }
      }
   }
}



# Combine Training --------------------------------------------------------

# Train Set 1: All samples
training <- rbind(data.train.lacunes, data.train.nonlacunes)
# Randomise and save
training <- training[sample(nrow(training)),]

