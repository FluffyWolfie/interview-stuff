/*
mpicc asgn2.c -o asgn2.out -lm
mpirun -np 13 --oversubscribe asgn2.out 3 4 60

mpirun -np x*y+1 --oversubscribe asgn2.out x y iter
command line -> x, y, iteration count
processes must be x*y+1
x and y = sensor network node grid

log_file.txt refreshes for each run
 */

#include <stdio.h>
#include <mpi.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>

typedef struct send_node_log {
    int rank, dim1, dim2;
    float lat, longi, mag;
} node_log;

typedef struct send_seismic_reading {
    int year, month, day, hour, minute, second;
    float lat, longi, magnitude, depth;
} seismic_reading;

// function prototype
int base_station(MPI_Comm master_comm, MPI_Comm slave_comm, int iterations);

void sensor_node(MPI_Comm master_comm, MPI_Comm slave_comm, int row, int col);

double distance_comp(double lat1, double lon1, double lat2, double lon2);

double deg2rad(double deg);

_Noreturn void *balloon_thread_func(void *pArg);

int generate_seismic_reading(int size);

void delete();

//void display();

void q_insert(seismic_reading);

#define MSG_TERMINATE 1
#define MSG_send_comm_time_taken 2
#define MSG_abnormal_send 3
#define MSG_time 4
#define MSG_notify 5
#define pi 3.14159265358979323846
#define MAX_SIZE 10
#define DISTANCE_THRESHOLD_IN_KM 180
#define MAGNITUDE_THRESHOLD 2.50
#define MAGNITUDE_DIFFERENCE_THRESHOLD 0.50


// init for queue global array
seismic_reading global_queue[MAX_SIZE];
int rear = -1;
int front = -1;

int main(int argc, char *argv[]) {
    /// init variables for main function
    int rank, size, provided, row, col, sentinel_time;
    MPI_Comm new_comm;

    /// MPI init, get rank and size for each process
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // get row and col from user
    if (argc == 4) {
        row = (int) strtol(argv[1], NULL, 10);
        col = (int) strtol(argv[2], NULL, 10);
        sentinel_time = (int) strtol(argv[3], NULL, 10);

    } else {
        printf("Invalid arguments");
        MPI_Finalize();
        return 0;
    }

    /// split comm world into two parts, rank0 = root(base station)
    MPI_Comm_split(MPI_COMM_WORLD, rank == 0, 0, &new_comm);
    if (rank == 0)
        // root exec base station function
        base_station(MPI_COMM_WORLD, new_comm, sentinel_time);
    else
        // slave exec sensor node grid function
        sensor_node(MPI_COMM_WORLD, new_comm, row, col);
    MPI_Finalize();
    return 0;
}

// base station function
int base_station(MPI_Comm master_comm, MPI_Comm slave_comm, int iterations) {
    // custom data type for sending info from node to base
    // node rank, node dims[0], node dims[1], lat, long, mag
    MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Datatype mpi_log;
    MPI_Aint offsets[6];
    int blocklengths[6] = {1, 1, 1, 1, 1, 1};
    offsets[0] = offsetof(node_log, rank);
    offsets[1] = offsetof(node_log, dim1);
    offsets[2] = offsetof(node_log, dim2);
    offsets[3] = offsetof(node_log, lat);
    offsets[4] = offsetof(node_log, longi);
    offsets[5] = offsetof(node_log, mag);
    MPI_Type_create_struct(6, blocklengths, offsets, types, &mpi_log);
    MPI_Type_commit(&mpi_log);

    int size, sensor_nodes;
    MPI_Status status;
    MPI_Comm_size(master_comm, &size);
    // sensor node count = total process count - 1
    sensor_nodes = size - 1;
//    printf("*** Base Station Initialized ***\n");

    // Initiate Thread
    pthread_t tid;
    pthread_create(&tid, 0, balloon_thread_func, &sensor_nodes);

    // Verify Slave Count
    char str[128];
    node_log receive_log[5];
//    printf("Sensor Nodes: %d\n", sensor_nodes);
    // Store distance between reporting node and adjacent nodes
    float distance[4];
    bool condition;

    // reset file
    FILE *fp;
    fp = fopen("log_file.txt", "w");

    int msg_rank;
    int iteration = 0;
    float dist;

    double total_comm_time = 0;
    int overall_total_messages = 0,total_matching_adjacent_nodes = 0;
    int total_inconclusive = 0, total_conclusive = 0;

    // Iterate x amount of times based on user input
    while (iteration < iterations) {
        printf("Iteration %d\n", iteration + 1);
        for (int i = 1; i < sensor_nodes + 1; i++) {
            // Receive a check, to validate whether base station should be receiving a report
            MPI_Recv(&condition, 1, MPI_CXX_BOOL, i, MSG_notify,
                     master_comm, &status);
//            printf(">>> Condition: %d\n", condition);
            // If a report is triggered
            if (condition) {
//                printf(">>> Condition Triggered\n");
                // We receive the report time
                MPI_Recv(str, 128, MPI_CHAR, i, MSG_time,
                         master_comm, &status);
//                printf("*** Received Time: %s", str);
                // base station receive the log information
                MPI_Recv(receive_log, 5, mpi_log, i, MSG_abnormal_send,
                         master_comm, &status);

                double comm_time_start, comm_time_end, comm_time;
                comm_time_end = MPI_Wtime();
                MPI_Recv(&comm_time_start, 1, MPI_DOUBLE, i, MSG_send_comm_time_taken,
                         master_comm, &status);
                comm_time = fabs(comm_time_end - comm_time_start);

//                printf("end -> %.9f, start -> %.9f\n", comm_time_end, comm_time_start);
//                printf("from node to base station time received -> %.9f\n", comm_time);

//                printf("*** Receiving Log File from Sensor Node With Rank: %d, %d, %d, %d, Reporting Rank: %d\n",
//                       receive_log[0].rank, receive_log[1].rank, receive_log[2].rank, receive_log[3].rank,
//                       receive_log[4].rank);
                // For each adjacent node we compute the distance (last node is always reporting node)
                for (int k = 0; k < 4; k++) {
                    // If its a positive number, it indicates a valid adjacent node
                    if (receive_log[k].rank >= 0) {
                        dist = distance_comp(receive_log[4].lat, receive_log[4].longi,
                                             receive_log[k].lat, receive_log[k].longi);
//                        printf("dist -> %.5f\n", dist);
                        distance[k] = dist;
                    }
                        // If it's negative, it means the adjacent nodes does not exist, and we initialize
                        // the distance to a negative value, signifying invalid distance
                    else {
                        distance[k] = -1;
                    }
                }
                // Debug
//                for (int k = 0; k < 4; k++) {
//                    if (distance[k] >= 0) {
//                        printf("Computed Distance: %.2f\n", distance[k]);
//                    }
//                }

                // Global Queue Checking
                // Iterate through MAX_SIZE of results produced by balloon
                // Check for a match
                int q_itr = 0;
                bool matched = false;
                float balloon_dist, balloon_mag;
                // Storing of that specific matched entry in balloon
                seismic_reading seis[1];
                // Set to 10, can change to MAX_SIZE
                while (q_itr < 10 && !matched) {
                    // If the entries in the balloon is valid and the distance is within our designated range
                    // AND the magnitude is within a designated range
                    // We consider a conclusive alert
                    if (global_queue[q_itr].year != 0
                        && distance_comp(global_queue[q_itr].lat, global_queue[q_itr].longi,
                                         receive_log[4].lat, receive_log[4].longi) <= DISTANCE_THRESHOLD_IN_KM

                        &&
                        fabsf(global_queue[q_itr].magnitude - receive_log[4].mag) <= MAGNITUDE_DIFFERENCE_THRESHOLD) {

                        balloon_dist = distance_comp(global_queue[q_itr].lat, global_queue[q_itr].longi,
                                                     receive_log[4].lat, receive_log[4].longi);

                        balloon_mag = fabsf(global_queue[q_itr].magnitude - receive_log[4].mag);
                        matched = true;
                        // Store this particular entry from balloon to be matched later for logging
                        seis[0] = global_queue[q_itr];

//                        printf("Computed Distance Between Balloon: %.2f\n", balloon_dist);
//                        printf("Computed Magnitude Between Balloon: %.2f\n", balloon_mag);
                    }
                    q_itr++;
                }
                // Logging for event type
                char alert_type[50] = "Inconclusive";
                // If both magnitude and distance within our range, we set it as conclusive
                if (matched) {
                    strcpy(alert_type, "Conclusive");
                }

                time_t t;
                char time_str[128];
                // Logged Time
                time(&t);
                sprintf(time_str, "%s", asctime(localtime(&t)));

                // Log File Components
                fp = fopen("log_file.txt", "a");

                fprintf(fp, "==================================================================\n");
                fprintf(fp, "Iteration: %d\n", iteration + 1);
                fprintf(fp, "Logged time: %s", time_str);
                fprintf(fp, "Alert reported time: %s", str);
                fprintf(fp, "Alert type: %s\n", alert_type);
                fprintf(fp, "\n");
                fprintf(fp, "%-30s%-40s%-60s\n", "Reporting Node", "Seismic Coord", "Magnitude");
                fprintf(fp, "%d(%d, %d%-24s(%.2f,%.2f%-27s%-60.2f\n",
                        receive_log[4].rank, receive_log[4].dim1, receive_log[4].dim2, ")",
                        receive_log[4].lat, receive_log[4].longi, ")", receive_log[4].mag);
                fprintf(fp, "\n");
                fprintf(fp, "%-30s%-20s%-20s%-20s%-20s\n", "Adjacent Nodes", "Seismic Coord", "Diff(Coord.km)",
                        "Magnitude", "Diff(Mag)");

                float adj_and_report_distance, adj_and_report_magnitude;
                int matching_adjacent_nodes = 0;
                // Difference in distance and magnitude from reporting node
                for (int l = 0; l < 4; l++) {
                    if (receive_log[l].rank >= 0 && receive_log[l].lat != 1000) {
                        adj_and_report_distance = distance_comp(receive_log[l].lat, receive_log[l].longi,
                                                                receive_log[4].lat, receive_log[4].longi);

                        adj_and_report_magnitude = fabsf(receive_log[l].mag - receive_log[4].mag);

                        if ((adj_and_report_distance <= DISTANCE_THRESHOLD_IN_KM) &&
                            (adj_and_report_magnitude <= MAGNITUDE_DIFFERENCE_THRESHOLD)) {
                            matching_adjacent_nodes += 1;
                        }
                        fprintf(fp, "%d(%d, %d%-24s(%.2f,%.2f%-7s%-20.2f%-20.2f%-10.2f\n", receive_log[l].rank,
                                receive_log[l].dim1, receive_log[l].dim2, ")", receive_log[l].lat, receive_log[l].longi,
                                ")", adj_and_report_distance, receive_log[l].mag, adj_and_report_magnitude);
                    }
                }

                int total_messages = 4;
                fprintf(fp, "\n");
                if (!strcmp(alert_type, "Inconclusive")) {
                    total_inconclusive++;
                    fprintf(fp, "This is an inconclusive alert. Thus, no matched entries with balloon seismic.\n\n");
                } else {
                    total_conclusive++;
                    fprintf(fp, "Balloon seismic reporting time: %d-%d-%d (H) %d, (M) %d, (S) %d\n",
                            seis[0].year, seis[0].month, seis[0].day, seis[0].hour, seis[0].minute, seis[0].second);
                    fprintf(fp, "Balloon seismic reporting Coord: (%.2f,%.2f)\n",
                            seis[0].lat, seis[0].longi);
                    fprintf(fp, "Balloon seismic reporting Coord Diff. with Reporting Node (km): %.2f\n",
                            balloon_dist);
                    fprintf(fp, "Balloon seismic reporting Magnitude: %.2f\n",
                            seis[0].magnitude);
                    fprintf(fp, "Balloon seismic reporting Magnitude Diff. with Reporting Node (km): %.2f\n\n",
                            balloon_mag);
                }

                fprintf(fp, "Communication Time (seconds): %.9f\n", comm_time);
                fprintf(fp, "Total Messages send between reporting node and base station: %d\n", total_messages);
                fprintf(fp, "Number of adjacent matches to reporting node: %d\n", matching_adjacent_nodes);
                fprintf(fp, "Coordinate difference threshold (km): %d\n", DISTANCE_THRESHOLD_IN_KM);
                fprintf(fp, "Magnitude difference threshold: %.2f\n", MAGNITUDE_DIFFERENCE_THRESHOLD);
                fprintf(fp, "Earthquake magnitude threshold: %.2f\n", MAGNITUDE_THRESHOLD);

                total_comm_time += comm_time;
                overall_total_messages += total_messages;
                total_matching_adjacent_nodes += matching_adjacent_nodes;

                fprintf(fp, "==================================================================\n");
                fprintf(fp, "==================================================================\n");
                fprintf(fp, "\n");
                fclose(fp);

            }
            condition = false;
        }
        iteration++;
    }

    fp = fopen("log_file.txt", "a");
    fprintf(fp, "==================================================================\n");
    fprintf(fp, "Number of messages passed throughout network: %d\n", overall_total_messages);
    fprintf(fp, "Number of alerts: %d\n", total_inconclusive + total_conclusive);
    fprintf(fp, "Total communication time: %.9f\n", total_comm_time);
    fprintf(fp, "==================================================================\n");
    fclose(fp);

    // Exiting
    int node_exit_cond = 1;
    MPI_Request exit_req_bs[1];
    for (int i = 0; i < size; i++) {
        // send termination message to sensor nodes
        MPI_Isend(&node_exit_cond, 1, MPI_INT, i, MSG_TERMINATE, MPI_COMM_WORLD, &exit_req_bs[0]);
    }
    printf(">>> Master terminating...\n");
    // send termination message to balloon thread
    pthread_cancel(tid);
    printf(">>> Thread exited!\n");
    printf(">>>Program terminated successfully.\n");
    return 0;
}

// sensor node function
void sensor_node(MPI_Comm master_comm, MPI_Comm slave_comm, int row, int col) {
    /// custom data type for sending info from node to base
    // node rank, node dims[0], node dims[1], lat, long, mag
    MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Datatype mpi_log;
    MPI_Aint offsets[6];
    int blocklengths[6] = {1, 1, 1, 1, 1, 1};
    offsets[0] = offsetof(node_log, rank);
    offsets[1] = offsetof(node_log, dim1);
    offsets[2] = offsetof(node_log, dim2);
    offsets[3] = offsetof(node_log, lat);
    offsets[4] = offsetof(node_log, longi);
    offsets[5] = offsetof(node_log, mag);
    MPI_Type_create_struct(6, blocklengths, offsets, types, &mpi_log);
    MPI_Type_commit(&mpi_log);

    int size, rank;
    int dims[2];
    /// get master communicator size, slave communicator size and sensor node rank
    MPI_Comm_size(slave_comm, &size);
    MPI_Comm_rank(slave_comm, &rank);

    /// if size doesn't add up to row*col, push error msg and exit
    if (size <= 0 || row * col != size) {
        if (rank == 0) {
            printf("Invalid size, %d * %d != %d\n", row, col, size);
            MPI_Finalize();
        }
    } else {
        /// set dimensions to row and col given
        dims[0] = row, dims[1] = col;
    }

    /// create cartesian topology
    MPI_Dims_create(size, 2, dims);
//    if (rank == 0) {
//        // debug
//        printf("Rank -> %d, Slave comm size -> %d, Grid dims -> %d * %d \n",
//               rank, size, dims[0], dims[1]);
//    }

    int wrap_arr[2] = {0, 0};
    int ierr = 0;
    MPI_Comm slave_communicator;
    // which communicator to split, number of dimensions, dimension array, wrap_around array,
    // reorder or not, address to store new communicator
    // wrap_around array = for each dimension, whether it does the calculation wraps around from
    // end to start like arr[-1] = the last element
    ierr = MPI_Cart_create(slave_comm, 2, dims, wrap_arr,
                           0, &slave_communicator);
    if (ierr != 0) {
        printf("Error creating cart -> %d", ierr);
    }

    int coords[2], cart_rank;
    /// find current process coordinates in the new 2D communicator
    // communicator to find in, the process rank, number of dimensions, array to store results in
    MPI_Cart_coords(slave_communicator, rank, 2, coords);
    // find current process's rank in the new 2d topology
    MPI_Cart_rank(slave_communicator, coords, &cart_rank);

    int left, right, up, down;
    /// get adjacent neighbours and save it to a variable
    // current communicator, check which dimension, displacement from origin, source process,
    // dest process
    // 0 = row or i, 1 = column or j
    MPI_Cart_shift(slave_communicator, 0, 1, &up, &down);
    MPI_Cart_shift(slave_communicator, 1, 1, &left, &right);

    // debug
//    printf("Local sensor node rank: %d. Cart rank: %d. Coord: (%d, %d). "
//           "Left: %d. Right: %d. Top: %d. Bottom: %d\n",
//           rank, cart_rank, coords[0], coords[1], left, right, up, down);

    fflush(stdout);
    int node_exit_cond;

    do {
        double start, end;
        start = MPI_Wtime();

        /// simulate sensor reading
        // YYYY MM DD HH MM SS Lat    Long   Magnitude Depth (km)
        // 2022 09 05 10 01 10 -15.36 167.50 4.75      5.25
        // ex: 3x3,3x3
        float latitude, longitude, magnitude, depth;
        unsigned int seed = time(NULL) * (cart_rank + 1);
        // latitude and longitude should be in a range of 1 for each row/col, stating from -16
        // think of each block, x and y-axis is 1x1 of lat and long
        latitude = (float) (-16.0 + coords[1]) + (float) rand_r(&seed) / (float) (RAND_MAX / 0.9);
        longitude = (float) (160.0 + coords[0]) + (float) rand_r(&seed) / (float) (RAND_MAX / 0.9);
        // https://stackoverflow.com/questions/70341989/weighted-random-float-number-with-single-target-and-chance-of-hitting-target
        // target = 0.2, strength = 1.0, target = most frequent random number, 0-1, strength is how
        // frequent will the random number be, 1-inf, check link for graphs
        // normal dist
        double target = 0.2, strength = 1.0, base, adjust, value;
        base = (float) rand_r(&seed) / (float) (RAND_MAX / 1.0);
        adjust = (float) rand_r(&seed) / (float) (RAND_MAX / 1.0);
        adjust = 1.0 - pow(1.0 - adjust, strength);
        value = ((float) (1.0 - adjust) * base) + (adjust * target);
        // value is in a range of 0-1, base value for gacha, *8 = magnitude is in a range of 0-8
        magnitude = (float) value * 8;
        // depth is in between 0-200
        depth = (float) value * 200;

        /// TODO:uncomment this for checking sensor node rank and such, printf might increase comp time
//        printf("Sensor rank -> %d, Lat -> %.2f, Long -> %.2f, Mag -> %.2f, Depth -> %.2f\n",
//               rank, latitude, longitude, magnitude, depth);

        bool trigger = false;
        if (magnitude > 2.5) {
            trigger = true;
        }

        /// send_data data to adjacent node, so up down left right, and receive data from adjacent node
        int target_node[4] = {left, right, up, down};
        bool receive_trigger[4] = {false, false, false, false};

        MPI_Request send_req[4];
        MPI_Status send_status[4];
        MPI_Request receive_req[4];
        MPI_Status receive_status[4];

        /// send and receive trigger boolean from every node
        for (int i = 0; i < 4; i++) {
            MPI_Isend(&trigger, 1, MPI_CXX_BOOL, target_node[i], 0,
                      slave_communicator, &send_req[i]);
            MPI_Irecv(&receive_trigger[i], 1, MPI_CXX_BOOL, target_node[i],
                      0, slave_communicator, &receive_req[i]);
        }
        MPI_Waitall(4, send_req, send_status);
        MPI_Waitall(4, receive_req, receive_status);

//        printf("Rank: %d, Boolean Values: %d, %d, %d, %d\n", rank, receive_trigger[0], receive_trigger[1],
//               receive_trigger[2], receive_trigger[3]);

        // init send data with -1 to represent uninitialized state, rank,dims not used here so left blank
        node_log send_data;
        send_data.mag = -1, send_data.lat = -1, send_data.longi = -1;
        MPI_Request send_req2[4];
        MPI_Status send_status2[4];
        int count = 0;

        // Check if any adjacent nodes have any abnormal magnitude values, if they requested for data,
        // send all important information to them. Else, do nothing.
        for (int i = 0; i < 4; i++) {
            if (receive_trigger[i]) {
                send_data.rank = rank;
                send_data.mag = magnitude, send_data.lat = latitude, send_data.longi = longitude;
                send_data.dim1 = coords[0], send_data.dim2 = coords[1];
                MPI_Isend(&send_data, 1, mpi_log, target_node[i], 0,
                          slave_communicator, &send_req2[count]);
                count++;
                //printf("Rank: %d sent information to Rank: %d\n", rank, target_node[i]);
            }
        }
        // Wait all possible requests to be sent.
        MPI_Waitall(count, send_req2, send_status2);


        // This section caters for receiving data if required.
        MPI_Request receive_req2[4];
        MPI_Status receive_status2[4];
        node_log receive_data[5];
        // Preparing receive data, initializing magnitude, latitude and longitude to values that are definitely out of range
        for (int i = 0; i < 5; i++) {
            receive_data[i].mag = -1, receive_data[i].lat = 1000, receive_data[i].longi = 1000;
        }

        // Base alert count
        int alert_count = 0;
        // If current node has abnormal magnitude, that means it must have sent a request previously
        // to its adjacent nodes. Thus, receive the information sent from its adjacent nodes.
        if (magnitude > 2.5) {
            for (int i = 0; i < 4; i++) {
                if (target_node[i] >= 0) {
                    MPI_Irecv(&receive_data[i], 1, mpi_log, target_node[i],
                              0, slave_communicator, &receive_req2[i]);
//                    printf("Rank: %d received information from Rank: %d\n", rank, target_node[i]);
                } else {
                    // Denote the request as received for adjacent nodes < 4
                    // Ensuring that it is a completed request
                    receive_req2[i] = MPI_REQUEST_NULL;
                }
            }
            // Wait for a possible of up to 4 adjacent requests
            MPI_Waitall(4, receive_req2, receive_status2);

            // Now we validate the information from adjacent nodes
            for (int i = 0; i < 4; i++) {
                // Precaution check to ensure that we do not receive invalid data
                // Additional checks to ensure that the magnitude is also > minimum earthquake threshold
                // Calculate if its within magnitude range as well
                if (receive_data[i].lat != 1000
                    && receive_data[i].longi != 1000
                    && receive_data[i].mag > 2.5
                    && fabsf(magnitude - receive_data[i].mag) <= MAGNITUDE_DIFFERENCE_THRESHOLD) {
                    // Compute distance for comparison
                    double distance_diff = distance_comp(latitude, longitude,
                                                         receive_data[i].lat, receive_data[i].longi);
                    // debug
                    // printf("Distance difference between %d and %d = %.2f\n", rank, receive_data[i].rank, distance_diff);
                    // If it's within the set threshold, we increment our alert count
                    if (distance_diff <= DISTANCE_THRESHOLD_IN_KM) {
                        alert_count++;
                    }
                }
            }
        }

        // If two or more alerts are detected
        if (alert_count > 1) {

            // The reporting node itself has to send its information as well
            node_log send_log;
            send_log.rank = rank, send_log.dim1 = coords[0], send_log.dim2 = coords[1],
            send_log.lat = latitude, send_log.longi = longitude, send_log.mag = magnitude;

            // If this node triggered the alert, this means that it must have previously
            // obtained information from its adjacent nodes, thus, we append it to the last
            // possible node to denote reporting node
            receive_data[4] = send_log;

            // Get date and time when an alert is detected
            time_t t;
            char str[128];
            time(&t);
            sprintf(str, "%s", asctime(localtime(&t)));

//            printf("Time -> %s", asctime(localtime(&t)));

            // send to base station
            // can integrate into node_info, but its only needed once so /shrug

            bool is_message = true;
            MPI_Send(&is_message, 1, MPI_CXX_BOOL, 0, MSG_notify, master_comm);
//            printf("Sending Time\n");
            MPI_Send(str, strlen(str) + 1, MPI_CHAR, 0, MSG_time, master_comm);
//            printf("Sending Log File to Base Station With Rank: %d, %d, %d, %d, Reporting Rank: %d\n",
//                   receive_log[0].rank, receive_log[1].rank, receive_log[2].rank, receive_log[3].rank,
//                   receive_log[4].rank);
            MPI_Send(receive_data, 5, mpi_log, 0, MSG_abnormal_send, master_comm);
            // send communication time taken from node to base station
            double comm_time_start = MPI_Wtime();
            MPI_Send(&comm_time_start, 1, MPI_DOUBLE, 0, MSG_send_comm_time_taken, master_comm);
        } else {
            bool is_message = false;
            MPI_Send(&is_message, 1, MPI_CXX_BOOL, 0, MSG_notify, master_comm);
        }
        end = MPI_Wtime();
//        printf("sleep time in seconds -> %.9f\n", (1 - (end - start)));
        // each iteration will run exactly 1 second, comp time + sleep = 1 sec
        usleep((useconds_t) ((1 - (end - start)) * 1e+6));

        int terminate = 0;
        MPI_Request exit_req[1];
        // attempts to receive termination signal from base station
        MPI_Irecv(&terminate, 1, MPI_INT, 0, MSG_TERMINATE, MPI_COMM_WORLD, &exit_req[0]);
        // test whether did base station sends a termination message
        MPI_Test(&exit_req[0], &node_exit_cond, MPI_STATUS_IGNORE);
        // printf("Exit Condition: %d\n", terminate);
        node_exit_cond = terminate;

    } while (node_exit_cond != 1);
    // free comm and exit
    printf(">>> Slave exiting...\n");
    MPI_Comm_free(&slave_communicator);
}


int generate_seismic_reading(int size) {
    // init variables and get local time
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    seismic_reading n1;

    unsigned int seed = time(NULL) * tm.tm_sec;
    // same methodology for generating readings as sensor nodes using normal distribution
    double target = 0.2, strength = 1.0, base, adjust, value;
    base = (float) rand_r(&seed) / (float) (RAND_MAX / 1.0);
    adjust = (float) rand_r(&seed) / (float) (RAND_MAX / 1.0);
    adjust = 1.0 - pow(1.0 - adjust, strength);
    value = ((float) (1.0 - adjust) * base) + (adjust * target);

    // construct and fill in values for the objects that goes to global shared array
    n1.year = tm.tm_year + 1900;
    n1.month = tm.tm_mon + 1;
    n1.day = tm.tm_mday;
    n1.hour = tm.tm_hour;
    n1.minute = tm.tm_min;
    n1.second = tm.tm_sec;
    // for lat and long, size of 3x3 means -16 to -13 range, vise versa
    n1.lat = (float) (-16.0) + (float) rand_r(&seed) / (float) (RAND_MAX / size);
    n1.longi = (float) (160.0) + (float) rand_r(&seed) / (float) (RAND_MAX / size);
    n1.magnitude = 2.5 + (float) value * 5.5;
    n1.depth = (float) value * 200;
    q_insert(n1);
//    display();
    return 0;
}

// balloon runs on base station thread
_Noreturn void *balloon_thread_func(void *pArg) {
    int matrix = *(int *) pArg;
    int ind = (int) sqrt(matrix);
    while (1) {
        double start, end;
        start = MPI_Wtime();

        generate_seismic_reading(ind);

        end = MPI_Wtime();
//        printf("balloon sleep time -> %.9f\n", 1 - (end - start));
        // same sleep method as sensor nodes, comp time + sleep time = 1 sec
        usleep((useconds_t) ((1 - (end - start)) * 1e+6));
    }
}


double distance_comp(double lat1, double lon1, double lat2, double lon2) {
    // in KM(kilometers)
    // code taken and simplified from link in specs
    double theta, dist;
    if ((lat1 == lat2) && (lon1 == lon2)) {
        return 0;
    } else {
        theta = lon1 - lon2;
        dist = sin(deg2rad(lat1)) * sin(deg2rad(lat2)) +
               cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * cos(deg2rad(theta));
        dist = acos(dist);
        dist = dist * 180 / pi * 60 * 1.1515 * 1.609344;
        return dist;
    }
}

double deg2rad(double deg) {
    return (deg * pi / 180);
}

void q_insert(seismic_reading log) {
    // queue insert function
    // if rear pointer is at the end of global array, remove and append
    if (rear == MAX_SIZE - 1) {
        delete();
        global_queue[rear] = log;
    } else {
        // increment rear pointer and append
        if (front == -1) {
            front = 0;
        }
        rear += 1;
        global_queue[rear] = log;
    }
}

void delete() {
    if (front == -1 || front > rear) {
        printf("Queue [Nothing to delete!] \n");
        return;
    } else {
        // overwrite first item and second becomes first
        for (int i = 0; i < rear; i++) {
            global_queue[i] = global_queue[i + 1];
        }
    }
}

//void display() {
//    if (front == -1) {
//        printf("Queue [Empty]");
//    } else {
//        printf("Queue Elements: \n");
//        for (int i = front; i <= rear; i++) {
//            printf("---Reading %d---\n", i);
//            printf("(Time) Y: %d M: %d D: %d H: %d M: %d S: %d \n",
//                   global_queue[i].year, global_queue[i].month, global_queue[i].day,
//                   global_queue[i].hour, global_queue[i].minute, global_queue[i].second);
//            printf("(Information)\n");
//            printf("Latitude: %.2f\n", global_queue[i].lat);
//            printf("Longitude: %.2f\n", global_queue[i].longi);
//            printf("Magnitude: %.2f\n", global_queue[i].magnitude);
//            printf("Depth: %.2f\n", global_queue[i].depth);
//        }
//        printf("\n");
//    }
//}