#!/bin/bash

# Use sacct to get energy data for a job
# https://slurm.schedmd.com/sacct.html

# Check if at least one job ID is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 jobid_file"
  exit 1
fi

jobid_file="$1"

# Check if the job ID file exists
if [ ! -f "$jobid_file" ]; then
  echo "File not found: $jobid_file"
  exit 1
fi

# Loop through each job ID in the file, skipping the header and table borders
tail -n +4 "$jobid_file" | while  IFS=',' read -r jobid total_energy; do
  # Trim whitespace from jobid
  jobid=$(echo "$jobid" | xargs)
  total_energy=$(echo "$total_energy" | xargs)
  # Use sacct to get energy data for the job
  # sacct -j 3315113 --format=JobID,ConsumedEnergy,JobName --noheader --parsable2 | grep '|extern'
  energy_data=$(sacct -j "$jobid" --format=JobID,JobName,ConsumedEnergy,ConsumedEnergyRaw --noheader --parsable2 | grep '|extern')
  
  # Check if sacct returned any data
  if [ -z "$energy_data" ]; then
    echo "{\"jobid\": \"$jobid\", \"energy\": null,\"energy_raw\": null, \"total_energy\" : null },"
  else
    # Extract the energy value
    energy=$(echo "$energy_data" | awk -F'|' '{print $3}')
    energy_raw=$(echo "$energy_data" | awk -F'|' '{print $4}')
    echo "{\"jobid\": \"$jobid\", \"energy\": \"$energy\", \"energy_raw\": \"$energy_raw\", \"total_energy\":\"$total_energy\"},"
  fi
done






