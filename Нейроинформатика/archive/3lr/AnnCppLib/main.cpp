#include <iostream>
#include <SFML/Graphics.hpp>
#include <vector>
#include "ANN_Lib.hpp"
#include <cmath>
#include <algorithm>


FILE* images = fopen("DigRecData/train-images.idx3-ubyte", "rb");
FILE* labels = fopen("DigRecData/train-labels.idx1-ubyte", "rb");

const int NImageTestsCount = 30;
const int NImageTestsItresPerOneSet = 1;
size_t flagIm = 0;
size_t SetCounter = 0;
size_t CurrentSet = 0; 

std::vector<std::vector<double>> testsNImage(60000);
std::vector<std::vector<double>> testsNImage_o(60000);

void PrepareNImageDataset() {
    int trash;
    fread(&trash, sizeof(int), 1, images);
    int n_of_images;
    fread(&n_of_images, sizeof(int), 1, images);
    int n_of_rows;
    fread(&n_of_rows, sizeof(int), 1, images);
    int n_of_columns;
    fread(&n_of_columns, sizeof(int), 1, images);

    fread(&trash, sizeof(int), 1, labels);
    fread(&trash, sizeof(int), 1, labels);

    unsigned char pixel;
    unsigned char iLab;
    for(int k = 0; k < 60000; k++) {
        testsNImage[k] = std::vector<double>(28 * 28);
        for(int i = 0; i < 28 * 28; i++) {
            fread(&pixel, 1, 1, images);
            testsNImage[k][i] = (double)pixel / 255;
        }
        testsNImage_o[k] = std::vector<double>(10, 0);
        fread(&iLab, 1, 1, labels);
        testsNImage_o[k][(int)iLab] = 1;
    }



}
std::vector<double> InputNImage() {
    return testsNImage[(CurrentSet % ((60000) / NImageTestsCount)) * NImageTestsCount + (flagIm % NImageTestsCount)];
    //return testsNImage[(flagIm % NImageTestsCount)];
}
std::vector<double> OutputNImage() {
    std::vector<double> out = testsNImage_o[(CurrentSet % ((60000) / NImageTestsCount)) * NImageTestsCount + (flagIm++ % NImageTestsCount)];
    //std::vector<double> out = testsNImage_o[(flagIm++ % NImageTestsCount)];
    SetCounter++;

    if (SetCounter == NImageTestsItresPerOneSet * NImageTestsCount) {
        SetCounter = 0;
        CurrentSet++;
    }
    return out;
}

FILE* imagesT = fopen("DigRecData/t10k-images.idx3-ubyte", "rb");
FILE* labelsT = fopen("DigRecData/t10k-labels.idx1-ubyte", "rb");

void PrepareNImageDatasetTest() {
    int trash;
    fread(&trash, sizeof(int), 1, imagesT);
    int n_of_images;
    fread(&n_of_images, sizeof(int), 1, imagesT);
    int n_of_rows;
    fread(&n_of_rows, sizeof(int), 1, imagesT);
    int n_of_columns;
    fread(&n_of_columns, sizeof(int), 1, imagesT);

    fread(&trash, sizeof(int), 1, labelsT);
    fread(&trash, sizeof(int), 1, labelsT);
}

std::vector<double> InputNImageTest() {
    std::vector<double> out(28 * 28);
    unsigned char pixel;
    for(int i = 0; i < 28 * 28; i++) {
        fread(&pixel, 1, 1, imagesT);
        out[i] = (double)pixel / 255;
    }
    return out;
}
int OutputNImageTest() {
    unsigned char i;
    fread(&i, 1, 1, labelsT);
    return (int)i;
}

void TestImagesManual(FeedforwardANN& ANN) {
    std::vector<double> res;

    sf::RenderWindow window(sf::VideoMode(28*10, 28*10), "Dig");
    window.setFramerateLimit(120);
    sf::Vector2i mouce_pos;

    
    std::vector<sf::RectangleShape> rectangles(28 * 28, sf::RectangleShape(sf::Vector2f(9.f, 9.f)));
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            rectangles[i + j * 28].setPosition(i*10, j*10);
        }
    }

    std::vector<double> dig_vec;
    std::vector<double> to_ann_pic(28 * 28);

    for(int i = 0; i < 28 * 28; i++) {
        rectangles[i].setFillColor(sf::Color(0, 0, 0));
    }

    while (window.isOpen()) {   

        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // clear the window with black color
        window.clear(sf::Color::Black);

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            mouce_pos = sf::Mouse::getPosition(window);
            if (mouce_pos.x > 30 && mouce_pos.x < 240 && mouce_pos.y > 70 && mouce_pos.y < 240) {
                if(rectangles[((int)(mouce_pos.x) / 10 - 1) + ((int)(mouce_pos.y) / 10) * 28].getFillColor().r == 0 ) {
                    rectangles[((int)(mouce_pos.x) / 10 - 1) + ((int)(mouce_pos.y) / 10) * 28].setFillColor(sf::Color(150, 150, 150));
                }
                if(rectangles[((int)(mouce_pos.x) / 10 + 1) + ((int)(mouce_pos.y) / 10) * 28].getFillColor().r == 0 ) {
                    rectangles[((int)(mouce_pos.x) / 10 + 1) + ((int)(mouce_pos.y) / 10) * 28].setFillColor(sf::Color(150, 150, 150));
                }
                if(rectangles[((int)(mouce_pos.x) / 10) + ((int)(mouce_pos.y) / 10 - 1) * 28].getFillColor().r == 0 ) {
                    rectangles[((int)(mouce_pos.x) / 10) + ((int)(mouce_pos.y) / 10 - 1) * 28].setFillColor(sf::Color(150, 150, 150));
                }
                if(rectangles[((int)(mouce_pos.x) / 10) + ((int)(mouce_pos.y) / 10 + 1) * 28].getFillColor().r == 0 ) {
                    rectangles[((int)(mouce_pos.x) / 10) + ((int)(mouce_pos.y) / 10 + 1) * 28].setFillColor(sf::Color(150, 150, 150));
                }
                rectangles[((int)(mouce_pos.x) / 10) + ((int)(mouce_pos.y) / 10) * 28].setFillColor(sf::Color(255, 255, 255));
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Enter)) {
            while(sf::Keyboard::isKeyPressed(sf::Keyboard::Enter));
            sf::Color im_c;
            for(int i = 0; i < 28 * 28; i++) {
                to_ann_pic[i] = (double)rectangles[i].getFillColor().r / 255;
            }
            res = ANN.process(to_ann_pic);
            std::pair<int, double> max(0, res[0]);
            for(int j = 0; j < res.size(); j++) {
                std::cout << j << ": " << res[j] << '\n';
                if (max.second < res[j]) {
                    max.second = res[j];
                    max.first = j;
                }
            }
            std::cout << "Guess: " << max.first << "\n\n";
            for(int i = 0; i < 28 * 28; i++) {
                rectangles[i].setFillColor(sf::Color(0, 0, 0));
            }

        }

        for(int i = 0; i < 28 * 28; i++) {
            window.draw(rectangles[i]);
        }

        
        window.display();

    }
    
}
void TestImages(FeedforwardANN& ANN) {
    std::vector<double> res;
    int data;
    int count = 0;
    int pass = 0;
    
    for (int i = 0; i < 10000; i++) {
        res = ANN.process(InputNImageTest());
        data = OutputNImageTest();
        std::pair<int, double> max(0, res[0]);
        for(int j = 0; j < res.size(); j++) {
            //std::cout << res[j] << '\n';
            if (max.second < res[j]) {
                max.second = res[j];
                max.first = j;
            }
        }
        count++;
        //std::cout << data << ' ' << max.first << '\n';
        if (data == max.first) {
            pass++;
        }
    }
    std::cout << "Test accuracy:"  << (double)pass/count * 100 << '\n';
}
void TestImagesGrapf(FeedforwardANN& ANN) {
    std::vector<double> res;
    int data;
    int count = 0;
    int pass = 0;

    sf::RenderWindow window(sf::VideoMode(28*10, 28*10), "Dig");
    
    std::vector<sf::RectangleShape> rectangles(28 * 28, sf::RectangleShape(sf::Vector2f(9.f, 9.f)));
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            rectangles[i + j * 28].setPosition(i*10, j*10);
        }
    }

    std::vector<double> dig_vec;
    // run the program as long as the window is open
    int data_count = 0;
    while (window.isOpen() && (data_count++ < 60000))
    {   

        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // clear the window with black color
        window.clear(sf::Color::Black);

        dig_vec = InputNImageTest();
        data = OutputNImageTest();

        

        res = ANN.process(dig_vec);
        std::pair<int, double> max(0, res[0]);
        for(int j = 0; j < res.size(); j++) {
            std::cout << j << ": " << res[j] << '\n';
            if (max.second < res[j]) {
                max.second = res[j];
                max.first = j;
            } 
        }
        std::cout << '\n';

        count++;
        std::cout << data << ' ' << max.first << '\n';
        if (data == max.first) {
            pass++;
        }

        std::cout << "Current accuracy:"  << (double)pass/count * 100 << '\n';
        std::cout << '\n';

        for(int i = 0; i < 28 * 28; i++) {
            if(data ==  max.first) {
                rectangles[i].setFillColor(sf::Color(dig_vec[i]*255, dig_vec[i]*255, dig_vec[i]*255));
            } else {
                rectangles[i].setFillColor(sf::Color(dig_vec[i]*255, 0, 0));
            }
        }

        for(int i = 0; i < 28 * 28; i++) {
            window.draw(rectangles[i]);
        }

        window.display();

        if(data ==  max.first) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }
        //getchar();
    }
}

double x_0 = 0;
double y_0 = 0;
double p = -0.3;
double x_t_l (double t){return pow(t, 2)/(2*p) + x_0;};
double y_t_l (double t){return t + y_0;};

double y_t (double t){return sin(2.5*pow(t, 2)- 5 * t);};
double x_t (double t){return t;};

std::vector<std::vector<std::vector<double>>> PrepareFuncData() {
    
    std::vector<std::vector<double>> inpX;
    std::vector<std::vector<double>> outY;

    
    for (double t = 0; t < 2.2 ; t+=0.02) {
        inpX.push_back({(x_t(t) - 0.5) / 3.0});
        outY.push_back({(y_t(t) + 1.5) / 3.0});
        // std::reverse(inpX.begin(), inpX.end());
        // std::reverse(outY.begin(), outY.end());
    }
    auto inpXc = inpX;
    auto outYc = outY;
    std::vector<int> ind;
    for (int i = 0; i < inpX.size(); i++) {
        ind.push_back(i);
    }
    std::random_shuffle (ind.begin(), ind.end());
    for (int i = 0; i < ind.size(); i++) {
        inpX[i] = inpXc[ind[i]];
        outY[i] = outYc[ind[i]];
    }

    return {inpX, outY};
}

int main() {
    srand(time(0));

    PrepareNImageDataset();
    PrepareNImageDatasetTest();

    auto data = PrepareFuncData();
    FeedforwardANN test({1, 5, 1}, 0.1);
    test.load("Autosave.ann");
    StacGradDesend(test, data[0] , data[1], 110 ,10000, false);
    FILE* fp_fitness = fopen("points.txt","w"); // for ploting

    for (int i = 0; i < data[0].size(); i++) {
        std::cout << data[0][i][0] << " " << data[1][i][0] << "\n";
        fprintf(fp_fitness,"%f %f\n", data[0][i][0], data[1][i][0]);
    }
    for (int i = 0; i < data[0].size(); i++) {
        fprintf(fp_fitness,"%f %f\n", data[0][i][0], test.process(data[0][i])[0]);
    }

    // FeedforwardANN test({28*28, 100, 10}, 0.1);

    // // test.load("97.99.ann");
    // // test.load("Autosave.ann");


    // StacGradDesend(test, testsNImage , testsNImage_o, 10, 1, false);
    // // StacGradDesendOld(test, InputNImage, OutputNImage, 100, NImageTestsCount, false);
    // // RandomHillClimbing(test, InputNImage, OutputNImage, 100, NImageTestsCount, NImageTestsItresPerOneSet, false);
    // test.save("Save.ann");


    // TestImages(test);
    //TestImagesGrapf(test);
    //TestImagesManual(test);
    
    return 0;
}

