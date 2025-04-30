#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <mpi.h>

using namespace std;

pair<vector<double>, double> multiply_matrix_by_vector_and_max(const vector<vector<double>>& matrix, const vector<double>& vec) {
    vector<double> result(matrix.size(), 0.0);
    double max_elem = 0.0;

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
        if (abs(result[i]) > abs(max_elem)) {
            max_elem = result[i];
        }
    }
    return { result, max_elem };
}

pair<vector<double>, double> normalize_and_compare(const vector<double>& new_vector,
    const vector<double>& old_vector,
    double max_elem) {
    vector<double> normalized(new_vector.size());
    double divergence = 0.0;

    for (size_t i = 0; i < new_vector.size(); ++i) {
        normalized[i] = new_vector[i] / max_elem;
        divergence += pow(normalized[i] - old_vector[i], 2);
    }

    return { normalized, divergence };
}

pair<vector<vector<double>>, size_t> read_input(const string& filename) {
    ifstream file(filename);
    if (!file) throw runtime_error("Cannot open file: " + filename);

    size_t n;
    file >> n;
    vector<vector<double>> matrix(n, vector<double>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            file >> matrix[i][j];
        }
    }

    return { matrix, n };
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string input_file_name = argv[1];
    const double epsilon = 0.001;
    vector<double> flat_matrix;
    vector<double> eigenvalue_vector;
    size_t n = 0;

    try {
        if (rank == 0) {
            auto [matrix, n_temp] = read_input(input_file_name);
            n = n_temp;
            eigenvalue_vector.resize(n, 1.0);

            flat_matrix.reserve(n * n);
            for (const auto& row : matrix) {
                flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
            }
        }

        MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        if (rank != 0) eigenvalue_vector.resize(n);
        MPI_Bcast(eigenvalue_vector.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int min_rows_per_process = n / size;
        int remainder = n % size;
        int local_rows = min_rows_per_process + (rank < remainder ? 1 : 0);

        vector<int> recv_counts(size), offsets(size);
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            recv_counts[r] = min_rows_per_process + (r < remainder ? 1 : 0);
            offsets[r] = offset;
            offset += recv_counts[r];
        }

        vector<int> counts_flat(size), displs_flat(size);
        if (rank == 0) {
            int offset_flat = 0;
            for (int r = 0; r < size; ++r) {
                counts_flat[r] = recv_counts[r] * n;
                displs_flat[r] = offset_flat;
                offset_flat += counts_flat[r];
            }
        }

        vector<double> local_matrix_data(local_rows * n);
        MPI_Scatterv(flat_matrix.data(), counts_flat.data(), displs_flat.data(), MPI_DOUBLE,
            local_matrix_data.data(), local_rows * n, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        vector<vector<double>> local_matrix(local_rows, vector<double>(n));
        for (int i = 0; i < local_rows; ++i) {
            copy(local_matrix_data.begin() + i * n,
                local_matrix_data.begin() + (i + 1) * n,
                local_matrix[i].begin());
        }

        double global_div = 1.0;
        double start = MPI_Wtime();

        while (global_div >= epsilon) {
            auto [local_result, local_max] = multiply_matrix_by_vector_and_max(local_matrix, eigenvalue_vector);


            double global_max;
            MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            auto [local_norm, local_div] = normalize_and_compare(
                local_result,
                vector<double>(eigenvalue_vector.begin() + offsets[rank],
                    eigenvalue_vector.begin() + offsets[rank] + local_rows),
                global_max
            );

            copy(local_norm.begin(), local_norm.end(),
                eigenvalue_vector.begin() + offsets[rank]);

            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                eigenvalue_vector.data(), recv_counts.data(), offsets.data(), MPI_DOUBLE,
                MPI_COMM_WORLD);

            MPI_Allreduce(&local_div, &global_div, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            double finish = MPI_Wtime();
            cout << "Eigenvector:";
            for (double val : eigenvalue_vector) cout << " " << val;
            cout << "\nTime: " << (finish - start) << " seconds.\n";
        }

    }
    catch (const exception& e) {
        if (rank == 0) cerr << "Error: " << e.what() << endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}