#include <iostream>
#include <string>
#include <thread>
#include <random>
#include <cmath>
#include <vector>

double sigm(double x) {
    return 1 / (1 +  exp(-(x)));
}

double sigm_der(double x) {
    return sigm(x)*(double)(1.0 - sigm(x));
}

class FeedforwardANN {
public:
    FeedforwardANN(std::vector<int> layers_struct, double learning_rate = 0.01, int thread_count = 8):delta(0,1), re(time(0)), sf(learning_rate) {
        layers.resize(layers_struct.size());
        layers[0].size = layers_struct[0];
        layers[0].nodes.resize(layers[0].size);
        threads.resize(thread_count);
        std::uniform_real_distribution<double> initial_rand(-1, 1);
        for(int i = 1; i < layers_struct.size(); i++) {
            layers[i].size = layers_struct[i];
            layers[i].nodes.resize(layers[i].size);
            layers[i].biases.resize(layers[i].size, 0);
            layers[i].weights.resize(layers[i].size);
            for(int k = 0; k < layers_struct[i]; k++) {
                layers[i].weights[k].resize(layers[i - 1].size, initial_rand(re));
            }
        }
    }
    FeedforwardANN(const FeedforwardANN& source): delta(0,1), re(time(0)), sf(source.sf) {
        threads.resize(threads.size());
        size = source.size;
        layers = source.layers;
    }
    void operator= (const FeedforwardANN& source) {
        size = source.size;
        layers = source.layers;
        threads.resize(source.threads.size());
    }

    double norm() {
        double sum = 0;
        for(int i = 1; i < layers.size(); i++) {
            for(int k = 0; k < layers[i].size; k++) {
                for(int j = 0; j < layers[i].weights[k].size(); j++) {
                    sum += pow(layers[i].weights[k][j], 2);
                }
            }
            for(int j = 0; j < layers[i].biases.size(); j++) {
                sum += pow(layers[i].biases[j], 2);
            }
        }
        return sqrt(sum);
    }

    void randomize() {
        double lower_bound = 0.0001;
        double upper_bound = 0.1;
        std::uniform_real_distribution<double> delta(lower_bound,upper_bound);
        std::default_random_engine re(time(0));
        for(int i = 1; i < layers.size(); i++) {
            for(int k = 0; k < layers[i].size; k++) {
                for(int j = 0; j < layers[i].weights[k].size(); j++) {
                    layers[i].weights[k][j] = delta(re);
                }
            }
            for(int j = 0; j < layers[i].biases.size(); j++) {
                layers[i].biases[j] = delta(re);
            }
        }
    }

    void mutate(double cost) {
        srand(clock() + time(0));
        double step_size = sf * fmax(cost, 1);
        // std::cauchy_distribution<double> delta(0,1);
        // std::default_random_engine re(time(0));
        for(int i = 1; i < layers.size(); i++) {
            for(int k = 0; k < layers[i].size; k++) {
                for(int j = 0; j < layers[i].weights[k].size(); j++) {
                    if (rand() % 30 == 0)
                        layers[i].weights[k][j] += step_size * delta(re);
                }
            }
            for(int j = 0; j < layers[i].biases.size(); j++) {
                if (rand() % 30 == 0)
                    layers[i].biases[j] += step_size * delta(re);
            }
        }
    }

    FeedforwardANN gradient(std::vector<double>& source) {
        FeedforwardANN grad(*this);
        int thread_counter = 0;
        int group_size;

        auto last = [&](int start, int end) {
            if (end > layers[layers.size() - 1].nodes.size()) {
                end = layers[layers.size() - 1].nodes.size();
            }
            for(int i = start; i < end; i++) {
                grad.layers[layers.size() - 1].nodes[i] = (double)2 * (layers[layers.size() - 1].nodes[i] - source[i]);
            }
        };

        auto weights_and_biases = [&](int layer_counter, int start, int end) {
            if (end > grad.layers[layer_counter].size) {
                end = grad.layers[layer_counter].size;
            }
            for(int j = start; j < end; j++) {
                for(int k = 0; k < grad.layers[layer_counter - 1].size; k++) {
                        // if(layers[layer_counter].nodes[j] > 0)
                        //     grad.layers[layer_counter].weights[j][k] = layers[layer_counter - 1].nodes[k] * grad.layers[layer_counter].nodes[j];
                        // else
                        //     grad.layers[layer_counter].weights[j][k] = 0;
                        grad.layers[layer_counter].weights[j][k] = layers[layer_counter - 1].nodes[k] * sigm_der(layers[layer_counter].nodes[j]) * grad.layers[layer_counter].nodes[j];
                    }
                    grad.layers[layer_counter].biases[j] = sigm_der(layers[layer_counter].nodes[j]) * grad.layers[layer_counter].nodes[j];
                    // if(layers[layer_counter].nodes[j] > 0)
                    //     grad.layers[layer_counter].biases[j] = grad.layers[layer_counter].nodes[j];
                    // else
                    //     grad.layers[layer_counter].biases[j] = 0;
            }
        };

        auto prev_nodes = [&](int layer_counter, int start, int end) {
            if (end > grad.layers[layer_counter - 1].size) {
                end = grad.layers[layer_counter - 1].size;
            }
            for(int j = start; j < end; j++) {
                grad.layers[layer_counter - 1].nodes[j] = 0;
                for(int k = 0; k < grad.layers[layer_counter].size; k++) {
                    // if(layers[layer_counter].nodes[k] > 0) {
                    //     grad.layers[layer_counter - 1].nodes[j] += grad.layers[layer_counter].nodes[k] * layers[layer_counter].weights[k][j];
                    // }
                    grad.layers[layer_counter - 1].nodes[j] += sigm_der(layers[layer_counter].nodes[k]) * grad.layers[layer_counter].nodes[k] * layers[layer_counter].weights[k][j];
                }
            } 
        };

        group_size = (int)ceil((double)layers[layers.size() - 1].nodes.size() / threads.size());
        for(int k = 0; k < layers[layers.size() - 1].nodes.size(); k += group_size) {
            threads[thread_counter++] = std::thread(last, k, k + group_size);
        }
        for(int k = 0; k < threads.size(); k++) {
            if (threads[k].joinable()) {
                threads[k].join();
            }
        }
        thread_counter = 0;

        for(int i = (layers.size() - 1); i > 0; i--){
            group_size = (int)ceil((double)grad.layers[i].size / threads.size());
            for(int j = 0; j < grad.layers[i].size; j += group_size) {
                threads[thread_counter++] = std::thread(weights_and_biases, i, j, j + group_size);
            }
            for(int k = 0; k < threads.size(); k++) {
                if (threads[k].joinable()) {
                    threads[k].join();
                }
            }
            thread_counter = 0;

            group_size = (int)ceil((double)grad.layers[i - 1].size / threads.size());
            for(int j = 0; j < grad.layers[i - 1].size; j += group_size) {
                threads[thread_counter++] = std::thread(prev_nodes, i, j, j + group_size);
            }
            for(int k = 0; k < threads.size(); k++) {
                if (threads[k].joinable()) {
                    threads[k].join();
                }
            }
            thread_counter = 0;
        }
        return grad;
    }

    void operator*= (double scale){
        for(int i = 1; i < layers.size(); i++){
            for(int j = 0; j < layers[i].size; j++) {
                layers[i].biases[j] *= scale;
                layers[i].nodes[j] *= scale;
                for(int k = 0; k < layers[i].weights[j].size(); k++) {
                    layers[i].weights[j][k] *= scale;
                }
            }
        }
    }

    void operator+= (const FeedforwardANN& second){
        for(int i = 1; i < layers.size(); i++){
            for(int j = 0; j < layers[i].size; j++) {
                layers[i].biases[j] += second.layers[i].biases[j];
                layers[i].nodes[j] += second.layers[i].biases[j];
                for(int k = 0; k < layers[i].weights[j].size(); k++) {
                    layers[i].weights[j][k] += second.layers[i].weights[j][k];
                }
            }
        }
    }

    int output_size(){
        return layers[layers.size() - 1].size;
    }

    void print(){
        for(int i = 1; i < layers.size(); i++) {
            std::cout << i << " layer" << '\n';
            std::cout << "waights" << '\n';
            for(int k = 0; k < layers[i].size; k++) {
                for(int j = 0; j < layers[i].weights[k].size(); j++) {
                    std::cout << layers[i].weights[k][j] << ' ';
                }
                std::cout << '\n';
            }
            std::cout << "biases" << '\n';
            for(int j = 0; j < layers[i].biases.size(); j++) {
                std::cout << layers[i].biases[j] << ' ';
            }
            std::cout <<"\nnodes" << '\n';
            for(int j = 0; j < layers[i].nodes.size(); j++) {
                std::cout << layers[i].nodes[j] << ' ';
            }
            std::cout << '\n';
        std::cout << '\n';
        }
    }
    
    std::vector<double> process(std::vector<double>& input) {
        layers[0].nodes = input;
        auto process_n_nods = [&](int layer_counter, int node_start, int node_end) {
            double ans = 0;
            if (node_end > layers[layer_counter].size)
                node_end = layers[layer_counter].size;
            for(int k = node_start; k < node_end; k++) {
                ans = 0;
                for(int i = 0; i < layers[layer_counter - 1].size; i++) {
                    ans += layers[layer_counter].weights[k][i] * layers[layer_counter - 1].nodes[i];
                }
                layers[layer_counter].nodes[k] = 1 / (1 +  exp(-(ans + layers[layer_counter].biases[k])));
                // layers[layer_counter].nodes[k] = fmax(0, ans + layers[layer_counter].biases[k]);
            }
            
        };

        threads.resize(8);
        int thread_counter = 0;
        int group_size;;
        for(int i = 1; i < layers.size(); i++) {
            group_size = (int)ceil((double)layers[i].size / threads.size());
            for(int k = 0; k < layers[i].size; k += group_size) {
                threads[thread_counter++] = std::thread(process_n_nods, i, k, k + group_size);
            }
            for(int k = 0; k < threads.size(); k++) {
                if (threads[k].joinable()) {
                    threads[k].join();
                }
            }
            thread_counter = 0;
        }


        return layers[layers.size() - 1].nodes;
    }

    std::vector<double> process(std::vector<double>&& input) {
        layers[0].nodes = input;
        auto process_n_nods = [&](int layer_counter, int node_start, int node_end) {
            double ans = 0;
            if (node_end > layers[layer_counter].size)
                node_end = layers[layer_counter].size;
            for(int k = node_start; k < node_end; k++) {
                ans = 0;
                for(int i = 0; i < layers[layer_counter - 1].size; i++) {
                    ans += layers[layer_counter].weights[k][i] * layers[layer_counter - 1].nodes[i];
                }
                layers[layer_counter].nodes[k] = 1 / (1 +  exp(-(ans + layers[layer_counter].biases[k])));
                // layers[layer_counter].nodes[k] = fmax(0, ans + layers[layer_counter].biases[k]);
            }
            
        };

        threads.resize(8);
        int thread_counter = 0;
        int group_size;;
        for(int i = 1; i < layers.size(); i++) {
            group_size = (int)ceil((double)layers[i].size / threads.size());
            for(int k = 0; k < layers[i].size; k += group_size) {
                threads[thread_counter++] = std::thread(process_n_nods, i, k, k + group_size);
            }
            for(int k = 0; k < threads.size(); k++) {
                if (threads[k].joinable()) {
                    threads[k].join();
                }
            }
            thread_counter = 0;
        }


        return layers[layers.size() - 1].nodes;
    }

    double learning_rate() {
        return sf;
    }
    void save(std::string path) {
        FILE* stor = fopen(path.c_str(), "wb");
        fwrite(&size, sizeof(size_t), 1, stor);
        int size_of = threads.size();
        fwrite(&size_of, sizeof(int), 1, stor);
        size_of = layers.size();
        fwrite(&size_of, sizeof(int), 1, stor);
        for(int i = 0; i < layers.size(); i++ ) {
            size_of = layers[i].weights.size();
            fwrite(&size_of, sizeof(int), 1, stor);
            for (int k = 0; k < layers[i].weights.size(); k++) {
                int size_of_w = layers[i].weights[k].size();
                fwrite(&size_of_w, sizeof(int), 1, stor);
                for (int j = 0; j < size_of_w; j++) {
                    fwrite(&layers[i].weights[k][j], sizeof(double), 1, stor);
                }
            }
            size_of = layers[i].biases.size();
            fwrite(&size_of, sizeof(int), 1, stor);
            for (int k = 0; k < size_of; k++) {
                fwrite(&layers[i].biases[k], sizeof(double), 1, stor);
            }
            size_of = layers[i].nodes.size();
            fwrite(&size_of, sizeof(int), 1, stor);
            for (int k = 0; k < size_of; k++) {
                fwrite(&layers[i].nodes[k], sizeof(double), 1, stor);
            }
            fwrite(&layers[i].size, sizeof(size_t), 1, stor);
        }
        fclose(stor);
    }
    void load(std::string path) {
        FILE* stor = fopen(path.c_str(), "rb");
        fread(&size, sizeof(size_t), 1, stor);
        int size_of;
        fread(&size_of, sizeof(int), 1, stor);
        threads = std::vector<std::thread>(size_of);
        fread(&size_of, sizeof(int), 1, stor);
        layers = std::vector<Layer>(size_of);
        for(int i = 0; i < layers.size(); i++ ) {
            fread(&size_of, sizeof(int), 1, stor);
            layers[i].weights = std::vector<std::vector<double>>(size_of);
            for (int k = 0; k < layers[i].weights.size(); k++) {
                int size_of_w;
                fread(&size_of_w, sizeof(int), 1, stor);
                layers[i].weights[k] = std::vector<double>(size_of_w);
                for (int j = 0; j < size_of_w; j++) {
                    fread(&layers[i].weights[k][j], sizeof(double), 1, stor);
                }
            }
            fread(&size_of, sizeof(int), 1, stor);
            layers[i].biases = std::vector<double>(size_of);
            for (int k = 0; k < size_of; k++) {
                fread(&layers[i].biases[k], sizeof(double), 1, stor);
            }
            fread(&size_of, sizeof(int), 1, stor);
            layers[i].nodes = std::vector<double>(size_of);
            for (int k = 0; k < size_of; k++) {
                fread(&layers[i].nodes[k], sizeof(double), 1, stor);
            }
            fread(&layers[i].size, sizeof(size_t), 1, stor);

        }
        fclose(stor);
    }

private:
    class Layer {
        public:
            std::vector<std::vector<double>> weights;
            std::vector<double> biases;
            std::vector<double> nodes;
            size_t size;
    };
    std::cauchy_distribution<double> delta;
    std::default_random_engine re;
    double sf;

    size_t size;
    std::vector<Layer> layers;
    std::vector<std::thread> threads;
};


template<class I, class O>
void RandomHillClimbing(FeedforwardANN& source,I input, O output, double expected_acc, int size_for_fitness = 100, int itres_per_one_set = -1, bool is_rand = true) {
    FeedforwardANN current(source);
    if (is_rand)
        current.randomize();
    FeedforwardANN step(current);

    std::vector<double> res;
    std::vector<double> data;

    double cost = 0;
    double step_cost = 0;

    for(int k = 0; k < size_for_fitness; k++) {
        res = current.process(input());
        data = output();
        for(int j = 0; j < res.size(); j++) {
            cost += pow(res[j] - data[j], 2);
        }
    }

    int count;
    int pass;

    int set_counter = 1;

    while(true) {
        step = current;
        step.mutate(cost / size_for_fitness);
        count = 0;
        pass = 0;

        step_cost = 0;
        for(int k = 0; k < size_for_fitness; k++) {
            res = step.process(input());
            data = output();
            for(int j = 0; j < res.size(); j++) {
                step_cost += pow(res[j] - data[j], 2);
            }
        }
        if (step_cost < cost) {
            current = step;
            cost = step_cost;
        }

        std::cout << '\n' << set_counter << "'s itter";
        std::cout << "Step Cost:" << step_cost << '\n';
        std::cout << "Curr Accuracy: " << 100 - (cost / (size_for_fitness * source.output_size())) * 100 << '\n';
        std::cout << "Step Accuracy: " << 100 - (step_cost / (size_for_fitness * source.output_size())) * 100 << '\n';
        

        if (expected_acc < (100 - (cost/ (size_for_fitness * source.output_size()) * 100))) {
            source = current;
            break;
        }
        if (itres_per_one_set > 0) {
            current.save("Autosave.ann");
            set_counter++;
            if (itres_per_one_set == set_counter) {
                cost = 0;
                for(int k = 0; k < size_for_fitness; k++) {
                    res = current.process(input());
                    data = output();
                    for(int j = 0; j < res.size(); j++) {
                        cost += pow(res[j] - data[j], 2);
                    }
                    
                }
                std::cout << "\n\nNew Set\n\n"; 
                set_counter = 1;
            } 
        }
    }
}

template<class F>
void FitnessRandomHillClimbing(FeedforwardANN& source,F fit_func, double expected_fitness, int size_for_fitness = 100, bool is_rand = true) {
    FeedforwardANN current(source);
    if (is_rand)
        current.randomize();
    FeedforwardANN step(current);

    std::vector<double> res;

    double fitness = 0;
    double step_fitness = 0;

    for(int k = 0; k < size_for_fitness; k++) {
        fitness += fit_func(current);
    }

    int set_counter = 1;

    while(true) {
        step = current;
        step.mutate(10 / (fitness + 1) + 1);
        //step.mutate(2);

        step_fitness = 0;
        for(int k = 0; k < size_for_fitness; k++) {
            step_fitness += fit_func(step);
        }
        if (step_fitness > fitness) {
            current = step;
            fitness = step_fitness;
        }

        std::cout << '\n' << set_counter << "'s itter";
        std::cout << "Curr Fitness: " << fitness << '\n';
        std::cout << "Step Fitness:" << step_fitness << '\n';
        

        if (expected_fitness < fitness) {
            source = current;
            break;
        }
        current.save("Autosave.ann");
        set_counter++;
    }
}

template<class I, class O>
void MultyRandomHillClimbing(FeedforwardANN& source,I input, O output, double expected_acc, int start_number, int size_for_fitness = 100) {
    std::vector<FeedforwardANN> current(start_number, source);
    std::vector<FeedforwardANN> step(start_number, source);
    //FeedforwardANN step(source);

    std::vector<double> res;
    std::vector<double> data;

    std::vector<double> cost(start_number, 0);
    std::vector<double> step_cost(start_number, 0);

    for(int i = 1; i < current.size(); i++) {
        current[i].randomize();
    }

    for(int i = 0; i < current.size(); i++) {
        for(int k = 0; k < size_for_fitness; k++) {
            res = current[i].process(input());
            data = output();
            for(int j = 0; j < res.size(); j++) {
                cost[i] += pow(res[j] - data[j], 2);
                //std::cout << pow(res[j] - data[j], 2) << '\n';
            }
        }
    }


    while(true) {
        std::string out_str;
        for(int i = 1; i < current.size(); i++) {
            step[i] = current[i];
            step[i].mutate(cost[i] / size_for_fitness);
            step_cost[i] = 0;
            for(int k = 0; k < size_for_fitness; k++) {
                res = step[i].process(input());
                data = output();
                for(int j = 0; j < res.size(); j++) {
                    step_cost[i] += pow(res[j] - data[j], 2);
                }
            }
            if (step_cost[i] < cost[i]) {
                current[i] = step[i];
                cost[i] = step_cost[i];
            }
            out_str += std::to_string(cost[i]) + '\n';
            
        }

        double min_cost = cost[0];
        for(int k = 0; k < cost.size(); k++) {
            if (cost[k] < min_cost) {
                min_cost = cost[k];
            }
        }
        std::cout << out_str << '\n';
        std::cout << min_cost << '\n';
        std::cout << "Best Accuracy: " << 100 - (min_cost / (size_for_fitness * source.output_size())) * 100 << '\n' << '\n';

        current[0].save("Autosave.ann");

        if (expected_acc < (100 - (min_cost/ (size_for_fitness * source.output_size()) * 100))) {
            std::pair<int, double> min_cost(0, cost[0]);
            for(int i = 1; i < current.size(); i++) {
                if(cost[i] <= min_cost.second) {
                    min_cost.first = i;
                    min_cost.second = cost[i];
                }
            }
            source = current[min_cost.first];
            break;
        }
    }
}

template<class I, class O>
void StacGradDesendOld(FeedforwardANN& source,I input, O output, double expected_acc, int size_for_fitness = 100, bool is_rand = true) {
    FeedforwardANN current(source);
    if (is_rand)
        current.randomize();
    FeedforwardANN step(source);

    FILE* fp_fitness = fopen("test.dat","w"); // for ploting

    std::vector<double> res;
    std::vector<double> data;

    double cost = 0;

    int set_counter = 1;

    double mov_avg;
    const int mov_avg_size = 100;
    std::vector<double> fit_history;

    while(true) {
        step *= 0;

        cost = 0;
        for(int k = 0; k < size_for_fitness; k++) {
            res = current.process(input());
            data = output();
            for(int j = 0; j < res.size(); j++) {
                cost += pow(res[j] - data[j], 2);
            }
            step += current.gradient(data);
        }
        step *= ((double)-1 / size_for_fitness) * source.learning_rate();
        current += step;


        std::cout << '\n' << set_counter << "'s set\n";
        std::cout << "Fitness:" << cost / size_for_fitness << '\n';
        // std::cout << "Accuracy: " << cost / (size_for_fitness * source.output_size())) * 100 << '\n';

        if (fit_history.size() < mov_avg_size) {
            fit_history.push_back(cost / size_for_fitness);
        } else {
            fit_history[(set_counter - 1) % mov_avg_size] = cost / size_for_fitness;
        }
        mov_avg = 0;
        for(double i: fit_history) {
            mov_avg += i;
        }
        mov_avg /= fit_history.size();

        std::cout << "Fitness moving avg:" << mov_avg << '\n';

        fprintf(fp_fitness,"%d\t%f\n", set_counter, mov_avg);
        
        

        if (expected_acc < (100 - (cost/ (size_for_fitness * source.output_size()) * 100)) || sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
            source = current;
            break;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
            source = current;
            std::cout << "\nSaved\n";
            while (sf::Keyboard::isKeyPressed(sf::Keyboard::S));
        }
        if (set_counter % 1000 == 0) {
            current.save("Autosave.ann");
        }
        set_counter++;
    }

    fclose(fp_fitness);
}


void StacGradDesend(FeedforwardANN& source,std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output, int batch_size = 10, int epochs = 1, bool is_rand = true) {
    FeedforwardANN current(source);
    if (is_rand)
        current.randomize();
    FeedforwardANN step(source);
    FeedforwardANN last_step(source);
    // FeedforwardANN d(source);
    // FeedforwardANN d_l(source);
    // FeedforwardANN grad(source);
    // FeedforwardANN grad_l(source);
    // double b;

    FILE* fp_fitness = fopen("test.dat","w"); // for ploting

    std::vector<double> res;
    std::vector<double> data;

    double cost = 0;

    int set_counter = 1;

    double mov_avg;
    const int mov_avg_size = 100;
    std::vector<double> fit_history;

    int batches_in_epoch = floor(input.size() / batch_size);

    for(int epoch_i = 0; epoch_i < epochs; epoch_i++) {
        last_step *= 0;
        std::cout << "Epoch " << epoch_i + 1 << '/' << epochs << '\n';
        for(int batch_i = 0; batch_i < batches_in_epoch; batch_i++) {
            step *= 0;

            cost = 0;
            for(int k = 0; k < batch_size; k++) {
                res = current.process(input[batch_i * batch_size + k]);
                data = output[batch_i * batch_size + k];
                for(int j = 0; j < res.size(); j++) {
                    cost += pow(res[j] - data[j], 2);
                }
                step += current.gradient(data);
            }
            
            // grad = step;
            // grad *= ((double)-1 / batch_size);
            // if (batch_i == 0) {
            //     grad_l = grad;
            // }
            // b = pow(grad.norm(), 2)/ pow(grad_l.norm(), 2);
            // if(b > 50)
            //     b = 50;
            // grad_l = grad;
            // if (batch_i == 0) {
            //     d = grad;
            // } else {
            //     d_l *= b;
            //     grad += d_l;
            //     d = grad;
            // }
            // d_l = d;
            // d *= source.learning_rate();
            // current += d;

            last_step *= 0.95;
            step *= ((double)-1 / batch_size) * source.learning_rate();
            step += last_step;
            last_step = step;
            current += step;

            

            if (fit_history.size() < mov_avg_size) {
                fit_history.push_back(cost / batch_size);
            } else {
                fit_history[(set_counter - 1) % mov_avg_size] = cost / batch_size;
            }
            mov_avg = 0;
            for(double i: fit_history) {
                mov_avg += i;
            }
            mov_avg /= fit_history.size();

            std::cout << batch_i + 1 << '/' << batches_in_epoch << "  " << std::fixed <<  (float)(batch_i + 1) / batches_in_epoch * 100 << "%\t loss:" <<  mov_avg << '\r' << std::flush;

            fprintf(fp_fitness,"%d\t%f\n", set_counter, mov_avg);
            
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                source = current;
                break;
            }
            if (set_counter % 1000 == 0) {
                current.save("Autosave.ann");
            }
            set_counter++;
        }
        std::cout << std::endl;
    }
    source = current;
    fclose(fp_fitness);
}
