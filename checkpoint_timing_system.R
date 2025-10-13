# Checkpoint and Timing System for Protein Benchmark Pipeline
# Created: October 13, 2025
# Purpose: Track timing and save intermediate results for reproducibility

initialize_checkpoint_system <- function(output_model_path, out_label) {
    checkpoint_dir <- paste0(output_model_path, "checkpoints/")
    dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
    
    timing_results <- list()
    pipeline_start <- Sys.time()
    
    session_info <- list(
        timestamp_start = pipeline_start,
        session_info = sessionInfo(),
        working_directory = getwd(),
        r_version = R.version.string
    )
    
    cat("="*60, "\n")
    cat("CHECKPOINT SYSTEM INITIALIZED\n")
    cat("="*60, "\n")
    cat("Checkpoint directory:", checkpoint_dir, "\n")
    cat("Pipeline started at:", as.character(pipeline_start), "\n")
    cat("="*60, "\n\n")
    
    return(list(
        checkpoint_dir = checkpoint_dir,
        timing_results = timing_results,
        pipeline_start = pipeline_start,
        session_info = session_info,
        out_label = out_label
    ))
}

# Checkpoint function - saves data and timing
save_checkpoint <- function(data, step_name, checkpoint_system, start_time, 
                           description = "", print_summary = TRUE) {
    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    checkpoint_data <- list(
        data = data,
        step_info = list(
            step_name = step_name,
            description = description,
            start_time = start_time,
            end_time = end_time,
            elapsed_seconds = elapsed,
            timestamp = as.character(end_time)
        ),
        session_state = list(
            working_directory = getwd(),
            memory_usage = if(exists("gc")) gc() else "gc() not available"
        )
    )
    
    # Save checkpoint file
    checkpoint_file <- paste0(checkpoint_system$checkpoint_dir, 
                             sprintf("%02d_%s_%s.rda", 
                                   length(checkpoint_system$timing_results) + 1,
                                   step_name, 
                                   checkpoint_system$out_label))
    
    save(checkpoint_data, file = checkpoint_file)
    
    # Update timing results
    checkpoint_system$timing_results[[step_name]] <- elapsed
    
    # Print progress
    if(print_summary) {
        total_elapsed <- as.numeric(difftime(end_time, checkpoint_system$pipeline_start, units = "secs"))
        cat(sprintf("[CHECKPOINT] %s\n", step_name))
        cat(sprintf("  Description: %s\n", ifelse(description != "", description, "No description")))
        cat(sprintf("  Step time: %.2f seconds (%.2f minutes)\n", elapsed, elapsed/60))
        cat(sprintf("  Total time: %.2f seconds (%.2f minutes)\n", total_elapsed, total_elapsed/60))
        cat(sprintf("  Saved: %s\n", basename(checkpoint_file)))
        cat(sprintf("  Memory: %s\n", 
                   ifelse(exists("gc"), paste(gc()[,2], collapse=" / "), "Memory info unavailable")))
        cat("-"*50, "\n\n")
    }
    
    return(checkpoint_system)
}

# Load checkpoint function (for reproducing steps)
load_checkpoint <- function(step_name, checkpoint_dir, out_label) {
    # Find checkpoint file by pattern
    pattern <- paste0("*_", step_name, "_", out_label, ".rda")
    checkpoint_files <- list.files(checkpoint_dir, pattern = pattern, full.names = TRUE)
    
    if(length(checkpoint_files) == 0) {
        stop(sprintf("Checkpoint not found for step '%s' with label '%s' in directory '%s'", 
                    step_name, out_label, checkpoint_dir))
    }
    
    if(length(checkpoint_files) > 1) {
        warning(sprintf("Multiple checkpoint files found for step '%s'. Using the first one: %s", 
                       step_name, basename(checkpoint_files[1])))
    }
    
    checkpoint_file <- checkpoint_files[1]
    cat(sprintf("[LOADING CHECKPOINT] %s\n", basename(checkpoint_file)))
    
    load(checkpoint_file)
    
    if(!exists("checkpoint_data")) {
        stop(sprintf("Invalid checkpoint file: %s", checkpoint_file))
    }
    
    cat(sprintf("  Step: %s\n", checkpoint_data$step_info$step_name))
    cat(sprintf("  Created: %s\n", checkpoint_data$step_info$timestamp))
    cat(sprintf("  Duration: %.2f seconds\n", checkpoint_data$step_info$elapsed_seconds))
    cat("-"*50, "\n")
    
    return(checkpoint_data$data)
}

# List available checkpoints
list_checkpoints <- function(checkpoint_dir, out_label = NULL) {
    if(is.null(out_label)) {
        pattern <- "*.rda"
    } else {
        pattern <- paste0("*_", out_label, ".rda")
    }
    
    checkpoint_files <- list.files(checkpoint_dir, pattern = pattern, full.names = FALSE)
    
    if(length(checkpoint_files) == 0) {
        cat("No checkpoints found in", checkpoint_dir, "\n")
        return(invisible(NULL))
    }
    
    cat("Available checkpoints:\n")
    cat("="*50, "\n")
    
    for(file in sort(checkpoint_files)) {
        # Extract step info from filename
        parts <- strsplit(tools::file_path_sans_ext(file), "_")[[1]]
        step_number <- parts[1]
        step_name <- paste(parts[2:(length(parts)-1)], collapse="_")
        
        cat(sprintf("%s: %s\n", step_number, step_name))
    }
    
    return(checkpoint_files)
}

# Generate timing report
generate_timing_report <- function(checkpoint_system, save_to_file = TRUE) {
    timing_results <- checkpoint_system$timing_results
    total_time <- sum(unlist(timing_results))
    pipeline_end <- Sys.time()
    actual_total <- as.numeric(difftime(pipeline_end, checkpoint_system$pipeline_start, units = "secs"))
    
    report_lines <- c()
    report_lines <- c(report_lines, "PIPELINE TIMING REPORT")
    report_lines <- c(report_lines, paste("="*60))
    report_lines <- c(report_lines, paste("Generated:", as.character(pipeline_end)))
    report_lines <- c(report_lines, paste("Pipeline started:", as.character(checkpoint_system$pipeline_start)))
    report_lines <- c(report_lines, paste("Pipeline ended:", as.character(pipeline_end)))
    report_lines <- c(report_lines, paste("Total actual time:", sprintf("%.2f seconds (%.2f minutes)", actual_total, actual_total/60)))
    report_lines <- c(report_lines, "")
    report_lines <- c(report_lines, "STEP-BY-STEP BREAKDOWN:")
    report_lines <- c(report_lines, paste("-"*60))
    
    for(step_name in names(timing_results)) {
        time_sec <- timing_results[[step_name]]
        percentage <- (time_sec / total_time) * 100
        report_lines <- c(report_lines, 
                         sprintf("%-30s: %8.2f sec (%5.1f%%) [%6.2f min]", 
                                step_name, time_sec, percentage, time_sec/60))
    }
    
    report_lines <- c(report_lines, paste("-"*60))
    report_lines <- c(report_lines, 
                     sprintf("%-30s: %8.2f sec (100.0%%) [%6.2f min]", 
                            "TOTAL MEASURED", total_time, total_time/60))
    
    cat(paste(report_lines, collapse = "\n"), "\n")
    
    if(save_to_file) {
        report_file <- paste0(checkpoint_system$checkpoint_dir, 
                             "timing_report_", checkpoint_system$out_label, ".txt")
        writeLines(report_lines, report_file)
        cat("\nTiming report saved to:", report_file, "\n")
        return(report_file)
    }
    
    return(report_lines)
}

# Generate reproduction script
generate_reproduction_script <- function(checkpoint_system) {
    script_file <- paste0(checkpoint_system$checkpoint_dir, 
                         "reproduce_pipeline_", checkpoint_system$out_label, ".R")
    
    script_lines <- c()
    script_lines <- c(script_lines, "# Pipeline Reproduction Script")
    script_lines <- c(script_lines, paste("# Generated on:", as.character(Sys.time())))
    script_lines <- c(script_lines, paste("# Original pipeline started:", as.character(checkpoint_system$pipeline_start)))
    script_lines <- c(script_lines, "")
    script_lines <- c(script_lines, "# Load checkpoint system")
    script_lines <- c(script_lines, "source('checkpoint_timing_system.R')")
    script_lines <- c(script_lines, "")
    script_lines <- c(script_lines, "# Set paths")
    script_lines <- c(script_lines, paste0("checkpoint_dir <- '", checkpoint_system$checkpoint_dir, "'"))
    script_lines <- c(script_lines, paste0("out_label <- '", checkpoint_system$out_label, "'"))
    script_lines <- c(script_lines, "")
    script_lines <- c(script_lines, "# Load each step:")
    
    step_counter <- 1
    for(step_name in names(checkpoint_system$timing_results)) {
        time_taken <- checkpoint_system$timing_results[[step_name]]
        var_name <- gsub("[^A-Za-z0-9]", "_", step_name)
        
        script_lines <- c(script_lines, "")
        script_lines <- c(script_lines, paste("#", "-"*50))
        script_lines <- c(script_lines, paste("#", sprintf("Step %d: %s (took %.2f seconds)", step_counter, step_name, time_taken)))
        script_lines <- c(script_lines, paste("#", "-"*50))
        script_lines <- c(script_lines, sprintf("data_%s <- load_checkpoint('%s', checkpoint_dir, out_label)", var_name, step_name))
        script_lines <- c(script_lines, sprintf("cat('Loaded step %d: %s\\n')", step_counter, step_name))
        
        step_counter <- step_counter + 1
    }
    
    script_lines <- c(script_lines, "")
    script_lines <- c(script_lines, "cat('All checkpoints loaded successfully!\\n')")
    
    writeLines(script_lines, script_file)
    cat("Reproduction script saved to:", script_file, "\n")
    return(script_file)
}

# Save final comprehensive summary
save_final_summary <- function(checkpoint_system, additional_info = list()) {
    pipeline_end <- Sys.time()
    total_time <- sum(unlist(checkpoint_system$timing_results))
    actual_total <- as.numeric(difftime(pipeline_end, checkpoint_system$pipeline_start, units = "secs"))
    
    summary_data <- list(
        timing_results = checkpoint_system$timing_results,
        pipeline_timing = list(
            start_time = checkpoint_system$pipeline_start,
            end_time = pipeline_end,
            total_measured_time = total_time,
            actual_total_time = actual_total,
            overhead_time = actual_total - total_time
        ),
        checkpoint_info = list(
            checkpoint_directory = checkpoint_system$checkpoint_dir,
            out_label = checkpoint_system$out_label,
            number_of_checkpoints = length(checkpoint_system$timing_results)
        ),
        session_info = checkpoint_system$session_info,
        additional_info = additional_info
    )
    
    # Save comprehensive summary
    summary_file <- paste0(checkpoint_system$checkpoint_dir, 
                          "pipeline_summary_", checkpoint_system$out_label, ".rda")
    save(summary_data, file = summary_file)
    
    cat("Final summary saved to:", summary_file, "\n")
    return(summary_data)
}

# Convenience function to time a block of code
time_block <- function(expr, step_name, checkpoint_system, description = "", save_result = TRUE) {
    start_time <- Sys.time()
    
    cat(sprintf("[STARTING] %s\n", step_name))
    if(description != "") {
        cat(sprintf("  %s\n", description))
    }
    
    # Execute the expression
    result <- expr
    
    if(save_result) {
        checkpoint_system <- save_checkpoint(result, step_name, checkpoint_system, start_time, description)
    } else {
        end_time <- Sys.time()
        elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
        checkpoint_system$timing_results[[step_name]] <- elapsed
        cat(sprintf("[COMPLETED] %s (%.2f seconds)\n", step_name, elapsed))
    }
    
    return(list(result = result, checkpoint_system = checkpoint_system))
}

cat("="*60, "\n")
cat("CHECKPOINT AND TIMING SYSTEM LOADED\n")
cat("="*60, "\n")
cat("Available functions:\n")
cat("  - initialize_checkpoint_system()\n")
cat("  - save_checkpoint()\n")
cat("  - load_checkpoint()\n")
cat("  - list_checkpoints()\n")
cat("  - generate_timing_report()\n")
cat("  - generate_reproduction_script()\n")
cat("  - save_final_summary()\n")
cat("  - time_block()\n")
cat("="*60, "\n\n")