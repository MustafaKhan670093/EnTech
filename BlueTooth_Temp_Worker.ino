#include <Wire.h>
#include <LiquidCrystal_I2C.h>
// initialize the library with the numbers of the interface pins

#define analogPin A0 //the thermistor attach to 
#define beta 3950 //the beta of the thermistor
#define resistance 10 //the value of the pull-up resistor

void setup()
{
  // set up the LCD's number of columns and rows: 
  Serial.begin(9600);
}

void loop()
{
  long a =1023 - analogRead(analogPin); //read thermistor value 
  Serial.print("Raw reading ");
  Serial.println(a); 
  //the calculating formula of temperature
  float tempC = beta /(log((1025.0 * 10 / a - 10) / 10) + beta / 298.0) - 273.0 + 41;
  float tempF = 1.8*tempC + 32.0;
  Serial.print("Celcius ");
  Serial.println(tempC); 
  Serial.println("");
  delay(1000);
}

 
